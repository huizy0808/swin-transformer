# ===== train_mean_teacher.py =====
"""
Mean Teacher + Swin-Base 微调 + dongmai 惩罚机制 + 分类报告 + TensorBoard 监控
"""
import os, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
import timm
from tqdm import tqdm
from multiprocessing import freeze_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

# 固定随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===== 配置 =====
DATA_DIR    = r"E:\\hzy\\data_remove\\train\\3-class\\train2"
IMG_SIZE    = 128
BATCH_SIZE  = 128
NUM_WORKERS = 4
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS  = 80
PATIENCE    = 8
EMA_DECAY   = 0.99
CONSISTENCY_WEIGHT = 1.0

# TensorBoard
writer = SummaryWriter(log_dir='runs/mean_teacher')

# 数据增强
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.5)
])
transform_val = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# 加载数据集
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform_train)
CLASS_NAMES = full_dataset.classes
labels = [label for _, label in full_dataset.samples]
train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=0.1, stratify=labels, random_state=seed)
train_set = Subset(full_dataset, train_idx)
val_set = Subset(copy.deepcopy(full_dataset), val_idx)
val_set.dataset.transform = transform_val

# 类别权重
class_counts = [labels.count(i) for i in range(len(CLASS_NAMES))]
total = sum(class_counts)
class_weights = [total/c for c in class_counts]
dongmai_idx = full_dataset.class_to_idx.get('dongmai')
if dongmai_idx is not None:
    class_weights[dongmai_idx] *= 1.5
loss_weights = torch.tensor(class_weights, device=DEVICE)
sample_weights = [class_weights[labels[i]] for i in train_idx]

# Dataloader
train_loader = DataLoader(train_set, BATCH_SIZE, sampler=WeightedRandomSampler(sample_weights, len(train_idx), replacement=True), num_workers=NUM_WORKERS)
val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# 定义损失（加入dongmai惩罚）
class PenalizedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, dongmai_idx=None, penalty_factor=3.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.dongmai_idx = dongmai_idx
        self.penalty_factor = penalty_factor
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
    def forward(self, input, target):
        logpt = -self.ce(input, target)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.dongmai_idx is not None:
            pred = input.argmax(1)
            penalty_mask = (target == self.dongmai_idx) & (pred != target)
            loss[penalty_mask] *= self.penalty_factor
        return loss.mean()

# 初始化模型
def create_model():
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=len(CLASS_NAMES), img_size=(IMG_SIZE, IMG_SIZE))
    for name, param in model.named_parameters():
        param.requires_grad = any(x in name for x in ['layers.2', 'layers.3', 'head'])
    return model

# 更新teacher参数
def update_ema(teacher_model, student_model, alpha):
    for ema_param, param in zip(teacher_model.parameters(), student_model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data * (1 - alpha))

# 主训练函数
def train_mean_teacher():
    student = create_model().to(DEVICE)
    teacher = copy.deepcopy(student).to(DEVICE)
    for p in teacher.parameters():
        p.requires_grad = False

    criterion = PenalizedFocalLoss(gamma=2.0, weight=loss_weights, dongmai_idx=dongmai_idx, penalty_factor=3.0)
    mse_loss = nn.MSELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, student.parameters()), lr=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=5e-4, steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS)
    scaler = GradScaler()

    best_auc, wait = 0, 0
    for epoch in range(1, NUM_EPOCHS+1):
        student.train()
        train_loss, correct = 0.0, 0
        for x, y in tqdm(train_loader, desc=f'Train Ep{epoch}'):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                stu_out = student(x)
                tea_out = teacher(x).detach()
                cls_loss = criterion(stu_out, y)
                cons_loss = mse_loss(torch.softmax(stu_out, dim=1), torch.softmax(tea_out, dim=1))
                loss = cls_loss + CONSISTENCY_WEIGHT * cons_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item() * x.size(0)
            correct += (stu_out.argmax(1) == y).sum().item()
            update_ema(teacher, student, EMA_DECAY)

        train_acc = correct / len(train_set)
        writer.add_scalar('train/loss', train_loss/len(train_set), epoch)
        writer.add_scalar('train/acc', train_acc, epoch)

        # 验证
        student.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = student(x)
                all_logits.append(logits.softmax(dim=1).cpu().numpy())
                all_labels.append(y.cpu().numpy())

        y_score = np.concatenate(all_logits)
        y_true = np.concatenate(all_labels)
        y_pred = y_score.argmax(1)
        y_true_bin = label_binarize(y_true, classes=[0,1,2])
        auc = roc_auc_score(y_true_bin, y_score, average='macro', multi_class='ovr')

        print(f"\nEpoch {epoch}: AUC={auc:.4f}")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
        writer.add_scalar('val/auc', auc, epoch)

        if auc > best_auc:
            best_auc = auc
            wait = 0
            torch.save(student.state_dict(), 'mean_teacher_best____2.pth')
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    writer.close()

if __name__ == '__main__':
    freeze_support()
    train_mean_teacher()

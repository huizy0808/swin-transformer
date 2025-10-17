
# -*- coding: utf-8 -*-
"""
Created on hzy
Modified: Two-class version mean-teacher swin transformer
"""
import os
import sys
import json
tk = None
# GUI only if needed
try:
    import tkinter as tk
    from tkinter import filedialog
    tk = tk
except ImportError:
    pass
from collections import Counter

import numpy as np
import torch
from PIL import Image
import timm
from torchvision import transforms
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
# Disable scientific notation
torch.set_printoptions(sci_mode=False)
np.random.seed(42)


def load_model(model_path, num_classes, device):
    """Load Swin Transformer model"""
    model = timm.create_model(
        'swin_base_patch4_window7_224',
        pretrained=False,
        num_classes=num_classes,
        img_size=(128, 128),
        window_size=7
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def select_directory(title="选择文件夹"):
    """GUI directory selector"""
    if tk is None:
        raise RuntimeError("tkinter not available")
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title)
    return path or None


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[设备状态] 使用 {device} 进行推理")

    # 1. select image folder
    print("\n[步骤1/3] 选择图像文件夹")
    imgs_root = select_directory("选择包含类别子文件夹的图像根目录")
    if not imgs_root:
        print("错误：必须选择有效图像目录")
        sys.exit()

    # 2. select model weights
    print("\n[步骤2/3] 选择模型权重文件")
    weights_path = filedialog.askopenfilename(
        title="选择模型权重文件",
        filetypes=[("PTH files", "*.pt"), ("PKL files", "*.pth")]
    )
    if not weights_path:
        print("错误：必须选择有效权重文件")
        sys.exit()

    # 3. select class index JSON
    print("\n[步骤3/3] 选择类别索引文件")
    json_path = filedialog.askopenfilename(
        title="选择类别索引 JSON 文件",
        filetypes=[("JSON files", "*.json")]
    )
    if not json_path:
        print("错误：必须选择有效 JSON 文件")
        sys.exit()

    # preprocessing
    data_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # collect images and labels
    img_paths, true_labels = [], []
    for cls in os.listdir(imgs_root):
        cls_dir = os.path.join(imgs_root, cls)
        if os.path.isdir(cls_dir):
            for nm in os.listdir(cls_dir):
                if nm.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_paths.append(os.path.join(cls_dir, nm))
                    true_labels.append(cls)

    # load class index
    with open(json_path, 'r', encoding='utf-8') as f:
        class_indict = json.load(f)
    class_names = [class_indict[str(i)] for i in range(len(class_indict))]
    class_to_idx = {v: int(k) for k, v in class_indict.items()}
    num_classes = len(class_names)

    try:
        model = load_model(weights_path, num_classes, device)
        print("\n[模型信息]")
        print(f"  · Backbone: Swin-base")
        print(f"  · 类别数: {num_classes}")
        print(f"  · 权重路径: {weights_path}")
    except Exception as e:
        print(f"\n模型加载失败: {e}")
        sys.exit()

    # inference
    batch_size = 8
    total = len(img_paths)
    pred_labels = []
    pred_probs = []

    print("\n[预测进度]")
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch = img_paths[i:i+batch_size]
            imgs = []
            for p in batch:
                try:
                    imgs.append(data_transform(Image.open(p).convert("RGB")))
                except Exception as e:
                    print(f"跳过损坏图像 {p}: {e}")
            if not imgs:
                continue
            inputs = torch.stack(imgs).to(device)
            out = model(inputs)
            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = torch.argmax(out, 1)
            pred_labels += [class_indict[str(int(idx))] for idx in preds]
            pred_probs += probs.tolist()
            prog = (i + len(imgs)) / total
            bar = '■' * int(prog * 30) + '□' * (30 - int(prog * 30))
            print(f"\r进度: {bar} {prog*100:.1f}%", end="")

    print("\n\n[统计结果]")
    for cls, cnt in Counter(pred_labels).most_common():
        print(f"▸ {cls:<15}: {cnt:>4} 张 ({cnt/len(pred_labels):.1%})")

    if len(true_labels) != len(pred_labels):
        print("\n[警告] 标签数量不匹配，无法计算评价指标")
        return

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_probs = np.array(pred_probs)

    # metrics
    acc = np.mean(true_labels == pred_labels)
    class_acc = {cls: None for cls in class_names}
    for cls in class_names:
        idxs = np.where(true_labels == cls)[0]
        if idxs.size:
            class_acc[cls] = np.mean(pred_labels[idxs] == cls)

    prec = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    rec = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

    # AUC
    true_idx = np.array([class_to_idx[c] for c in true_labels])
    if num_classes == 2:
        # use positive class = index 1
        auc = roc_auc_score(true_idx, pred_probs[:, 1])
    else:
        true_bin = label_binarize(true_idx, classes=list(range(num_classes)))
        auc = roc_auc_score(true_bin, pred_probs, average='weighted', multi_class='ovr')

    print("\n[原始指标]")
    print(f"▸ Overall Accuracy: {acc:.3f}")
    for cls in class_names:
        acc_i = class_acc[cls]
        print(f"   · {cls} Accuracy: {acc_i:.3f}" if acc_i is not None else f"   · {cls} Accuracy: N/A")
    print(f"▸ Precision (weighted): {prec:.3f}")
    print(f"▸ Recall (weighted):    {rec:.3f}")
    print(f"▸ F1 Score (weighted):  {f1:.3f}")
    print(f"▸ AUC Score:            {auc:.3f}")

    # bootstrap CI
    B = 1000
    rng = np.random.default_rng(42)
    n = len(true_labels)
    boot = { 'acc': np.empty(B), 'prec': np.empty(B), 'rec': np.empty(B), 'f1': np.empty(B), 'auc': np.empty(B) }
    boot_cls = {cls: np.empty(B) for cls in class_names}

    for b in range(B):
        idx = rng.choice(n, size=n, replace=True)
        y_t = true_labels[idx]; y_p = pred_labels[idx]; y_prob = pred_probs[idx]
        boot['acc'][b] = np.mean(y_t == y_p)
        if num_classes == 2:
            boot['auc'][b] = roc_auc_score([class_to_idx[c] for c in y_t], y_prob[:,1])
            boot['prec'][b] = precision_score(y_t, y_p, average='weighted', zero_division=0)
            boot['rec'][b]  = recall_score(y_t, y_p, average='weighted', zero_division=0)
            boot['f1'][b]   = f1_score(y_t, y_p, average='weighted', zero_division=0)
        else:
            y_t_idx = np.array([class_to_idx[c] for c in y_t])
            y_t_bin = label_binarize(y_t_idx, classes=list(range(num_classes)))
            boot['auc'][b] = roc_auc_score(y_t_bin, y_prob, average='weighted', multi_class='ovr')
            boot['prec'][b] = precision_score(y_t, y_p, average='weighted', zero_division=0)
            boot['rec'][b]  = recall_score(y_t, y_p, average='weighted', zero_division=0)
            boot['f1'][b]   = f1_score(y_t, y_p, average='weighted', zero_division=0)
        for cls in class_names:
            ids = np.where(y_t == cls)[0]
            boot_cls[cls][b] = np.mean(y_p[ids] == cls) if ids.size else np.nan

    def ci(arr):
        clean = arr[~np.isnan(arr)]
        return np.percentile(clean, 2.5), np.percentile(clean, 97.5)

    acc_ci = ci(boot['acc'])
    prec_ci = ci(boot['prec'])
    rec_ci = ci(boot['rec'])
    f1_ci = ci(boot['f1'])
    auc_ci = ci(boot['auc'])
    cls_ci = {cls: ci(boot_cls[cls]) for cls in class_names}

    print("\n[95% 置信区间]")
    print(f"▸ Overall Accuracy CI: [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]")
    for cls in class_names:
        lo, hi = cls_ci[cls]
        print(f"   · {cls} Accuracy CI: [{lo:.3f}, {hi:.3f}]")
    print(f"▸ Precision CI: [{prec_ci[0]:.3f}, {prec_ci[1]:.3f}]")
    print(f"▸ Recall    CI: [{rec_ci[0]:.3f}, {rec_ci[1]:.3f}]")
    print(f"▸ F1 Score  CI: [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]")
    print(f"▸ AUC       CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")

    # # ROC curve
    # plt.figure(figsize=(7,7))
    # if num_classes == 2:
    #     fpr, tpr, _ = roc_curve(true_idx, pred_probs[:,1])
    #     plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    # else:
    #     for i,cls in enumerate(class_names):
    #         fpr, tpr, _ = roc_curve(true_bin[:,i], pred_probs[:,i])
    #         plt.plot(fpr, tpr, lw=2, label=f"ROC {cls} (AUC = {roc_auc_score(true_bin[:,i], pred_probs[:,i]):.3f})")
    #     plt.plot([0,1],[0,1],linestyle="--", color='gray', label='Random')
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve")
    # plt.legend(loc="lower right")
    # plt.grid(alpha=0.3)
    # plt.show()
    plt.figure(figsize=(7,7))
    if num_classes == 2:
       fpr, tpr, _ = roc_curve(true_idx, pred_probs[:,1])
       plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
       # 新增：绘制代表随机预测的虚线
       plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")  
    else:
       for i,cls in enumerate(class_names):
          fpr, tpr, _ = roc_curve(true_bin[:,i], pred_probs[:,i])
          plt.plot(fpr, tpr, lw=2, label=f"ROC {cls} (AUC = {roc_auc_score(true_bin[:,i], pred_probs[:,i]):.3f})")
       plt.plot([0,1],[0,1],linestyle="--", color='gray', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

    # confusion matrix with percentages
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
    row_sum = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_perc = cm / row_sum
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cnt = cm[i,j]
            txt = f"{cnt}\n({cm_perc[i,j]:.1%})" if row_sum[i,0] else f"{cnt}\n(–)"
            ax.text(j, i, txt, ha='center', va='center', color='white' if im.norm(cnt)>0.5 else 'black')
    ax.set_xticks(range(num_classes)); ax.set_xticklabels(class_names)
    ax.set_yticks(range(num_classes)); ax.set_yticklabels(class_names)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title("Confusion Matrix (Count & %)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    # === 打印预测错误的图像名称 ===
    print("\n[错分图像列表]")
    mismatch_count = 0
    for path, true, pred, prob in zip(img_paths, true_labels, pred_labels, pred_probs):
        if true != pred:
            mismatch_count += 1
            prob_dict = {cls: f"{p:.3f}" for cls, p in zip(class_names, prob)}
            print(f"✘ {os.path.basename(path)} | True: {true:<8} | Pred: {pred:<8} | Prob: {prob_dict}")
    print(f"\n共错分 {mismatch_count} 张图像")
    
    
if __name__ == '__main__':
    main()

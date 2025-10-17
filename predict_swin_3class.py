# -*- coding: utf-8 -*-
"""
Created on hzy
Modified: 增加各参数的 95% 置信区间计算、多分类 ROC 曲线绘制，以及混淆矩阵中显示每行百分比
"""
import os
import sys
import json
import tkinter as tk
from tkinter import filedialog
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
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 禁用科学计数法显示
torch.set_printoptions(sci_mode=False)
np.random.seed(42)

def load_model(model_path, num_classes, device):
    """加载 Swin Transformer 模型，与训练时保持一致"""
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
    """GUI 目录选择对话框"""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title)
    return path if path else None

def main():
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[设备状态] 使用 {device} 进行推理")

    # 1. 选择图像文件夹
    print("\n[步骤1/3] 选择图像文件夹")
    imgs_root = select_directory("选择包含类别子文件夹的图像根目录")
    if not imgs_root:
        print("错误：必须选择有效图像目录")
        sys.exit()

    # 2. 选择模型权重
    print("\n[步骤2/3] 选择模型权重文件")
    weights_path = filedialog.askopenfilename(
        title="选择模型权重文件",
        filetypes=[("PTH files", "*.pt"), ("PKL files", "*.pth")]
    )
    if not weights_path:
        print("错误：必须选择有效权重文件")
        sys.exit()

    # 3. 选择类别索引 JSON
    print("\n[步骤3/3] 选择类别索引文件")
    json_path = filedialog.askopenfilename(
        title="选择类别索引 JSON 文件",
        filetypes=[("JSON files", "*.json")]
    )
    if not json_path:
        print("错误：必须选择有效 JSON 文件")
        sys.exit()

    # 4. 数据预处理 —— Swin Transformer 通常使用 224×224 输入
    data_transform = transforms.Compose([
        transforms.Resize(int(128 * 1.14)),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 5. 收集所有图片路径和真实标签
    img_paths, true_labels = [], []
    for class_name in os.listdir(imgs_root):
        class_dir = os.path.join(imgs_root, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_paths.append(os.path.join(class_dir, img_name))
                    true_labels.append(class_name)

    # 6. 加载类别索引
    with open(json_path, 'r', encoding='utf-8') as f:
        class_indict = json.load(f)
    class_names = [class_indict[str(i)] for i in range(len(class_indict))]
    class_to_idx = {v: int(k) for k, v in class_indict.items()}

    # 7. 初始化模型
    try:
        model = load_model(weights_path, len(class_indict), device)
        print("\n[模型信息]")
        print(f"  · Backbone: Swin-base")
        print(f"  · 类别数: {len(class_indict)}")
        print(f"  · 权重路径: {weights_path}")
    except Exception as e:
        print(f"\n模型加载失败: {e}")
        sys.exit()

    # 8. 批量预测，并收集预测标签与预测概率
    batch_size = 8
    total = len(img_paths)
    pred_labels = []
    pred_probs = []  # 存放每张图对各类别的 softmax 概率，形状 (N, num_classes)

    print("\n[预测进度]")
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_paths = img_paths[i:i + batch_size]
            batch_imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_imgs.append(data_transform(img))
                except Exception as e:
                    print(f"跳过损坏图像 {p}: {e}")
            if not batch_imgs:
                continue

            inputs = torch.stack(batch_imgs).to(device)
            outputs = model(inputs)  # shape = [batch_size, num_classes]
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            _, preds = torch.max(outputs, 1)
            batch_preds = [class_indict[str(idx.item())] for idx in preds]

            pred_labels.extend(batch_preds)
            pred_probs.extend(probs.tolist())

            # 进度条
            prog = (i + len(batch_imgs)) / total
            bar = '■' * int(prog * 30) + '□' * (30 - int(prog * 30))
            print(f"\r进度: {bar} {prog * 100:.1f}%", end="")

    # 9. 统计结果
    print("\n\n[统计结果]")
    cnt = Counter(pred_labels)
    for cls, c in cnt.most_common():
        print(f"▸ {cls:<15}: {c:>4} 张 ({c/len(pred_labels):.1%})")

    # 10. 评价指标 & 置信区间 & ROC & 混淆矩阵
    if len(true_labels) == len(pred_labels):
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        pred_probs = np.array(pred_probs)  # shape = (N, num_classes)
        n_total = len(true_labels)

        # 10.1 计算原始指标
        acc = np.mean(true_labels == pred_labels)
        # 每类准确率
        class_acc = {}
        for cls in class_names:
            idxs = np.where(true_labels == cls)[0]
            if idxs.size == 0:
                class_acc[cls] = None
            else:
                class_acc[cls] = np.mean(pred_labels[idxs] == cls)

        prec = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
        rec = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

        # 多分类 AUC (One-vs-Rest)
        true_idx = np.array([class_to_idx[c] for c in true_labels])
        true_bin = label_binarize(true_idx, classes=list(range(len(class_names))))  # shape = (N, C)
        auc = roc_auc_score(true_bin, pred_probs, average='weighted', multi_class='ovr')

        print(f"\n[原始指标]")
        print(f"▸ Overall Accuracy: {acc:.3f}")
        for cls in class_names:
            acc_i = class_acc[cls]
            if acc_i is not None:
                print(f"   · {cls} Accuracy: {acc_i:.3f}")
            else:
                print(f"   · {cls} Accuracy: N/A")
        print(f"▸ Precision (weighted): {prec:.3f}")
        print(f"▸ Recall (weighted):    {rec:.3f}")
        print(f"▸ F1 Score (weighted):  {f1:.3f}")
        print(f"▸ AUC (OvR weighted):   {auc:.3f}")
        
        
        
        # === 微观 / 宏观准确率（按 WSI 和类别） ===
        from collections import defaultdict

        def extract_wsi_id(path):
            """从图像路径提取 WSI 编号：xxx(xx).jpg -> xxx"""
            name = os.path.basename(path)
            if '(' in name:
                return name.split('(')[0]
            return os.path.splitext(name)[0]

        # 收集每个WSI的信息
        wsi_info = {}  # {wsi_id: {'true': [...], 'pred': [...]} }
        for path, true, pred in zip(img_paths, true_labels, pred_labels):
            wsi_id = extract_wsi_id(path)
            if wsi_id not in wsi_info:
                wsi_info[wsi_id] = {'true': [], 'pred': []}
            wsi_info[wsi_id]['true'].append(true)
            wsi_info[wsi_id]['pred'].append(pred)

        # 统计每个WSI的预测情况
        correct_wsis = 0
        micro_acc_list = []
        category_wsi_map = defaultdict(list)

        print(f"\n[WSI 级准确率明细]")
        print(f"{'WSI编号':<15}{'真实类别':<10}{'正确/总数':<12}{'准确率':<10}{'是否预测正确'}")

        for wsi_id, info in wsi_info.items():
            true_list = info['true']
            pred_list = info['pred']
            total = len(true_list)
            correct = sum([t == p for t, p in zip(true_list, pred_list)])
            acc = correct / total
            micro_acc_list.append(acc)

            # 每个WSI的真实标签采用多数投票
            true_major = Counter(true_list).most_common(1)[0][0]
            category_wsi_map[true_major].append(acc)

            # 判断是否预测正确
            is_correct = acc > 0.5
            if is_correct:
                correct_wsis += 1

            print(f"{wsi_id:<15}{true_major:<10}{correct}/{total:<12}{acc:.3f}     {'√' if is_correct else '×'}")

        # 总体微观 / 宏观准确率
        micro_acc = np.mean(micro_acc_list)
        macro_acc = correct_wsis / len(wsi_info)

        print(f"\n[总体统计]")
        print(f"▸ Micro Accuracy (图块平均): {micro_acc:.3f}")
        print(f"▸ Macro Accuracy (WSI >50%): {macro_acc:.3f}")

        # 按类别统计微观 / 宏观准确率
        print(f"\n[按类别统计准确率]")
        for cls in class_names:
            acc_list = category_wsi_map.get(cls, [])
            if len(acc_list) == 0:
                print(f"▸ {cls:<10}: 无数据")
                continue
            micro_c = np.mean(acc_list)
            macro_c = sum([a > 0.5 for a in acc_list]) / len(acc_list)
            print(f"▸ {cls:<10}: Micro = {micro_c:.3f} | Macro = {macro_c:.3f} | 共 {len(acc_list)} 张WSI")





        # ===== 10.2 Bootstrap 计算 95% 置信区间 =====
        B = 1000
        rng = np.random.default_rng(42)
        boot_acc = np.empty(B)
        boot_precision = np.empty(B)
        boot_recall = np.empty(B)
        boot_f1 = np.empty(B)
        boot_auc = np.empty(B)
        boot_class_acc = {cls: np.empty(B) for cls in class_names}

        indices = np.arange(n_total)
        for b in range(B):
            sample_idx = rng.choice(indices, size=n_total, replace=True)
            y_true_b = true_labels[sample_idx]
            y_pred_b = pred_labels[sample_idx]
            y_prob_b = pred_probs[sample_idx]

            # 整体准确率
            acc_b = np.mean(y_true_b == y_pred_b)
            boot_acc[b] = acc_b

            # Precision / Recall / F1
            if len(np.unique(y_true_b)) == len(class_names):
                precision_b = precision_score(y_true_b, y_pred_b, average='weighted', zero_division=0)
                recall_b = recall_score(y_true_b, y_pred_b, average='weighted', zero_division=0)
                f1_b = f1_score(y_true_b, y_pred_b, average='weighted', zero_division=0)
                # AUC
                y_true_b_idx = np.array([class_to_idx[c] for c in y_true_b])
                y_true_b_bin = label_binarize(y_true_b_idx, classes=list(range(len(class_names))))
                try:
                    auc_b = roc_auc_score(y_true_b_bin, y_prob_b, average='weighted', multi_class='ovr')
                except ValueError:
                    auc_b = np.nan
            else:
                precision_b = np.nan
                recall_b = np.nan
                f1_b = np.nan
                auc_b = np.nan

            boot_precision[b] = precision_b
            boot_recall[b] = recall_b
            boot_f1[b] = f1_b
            boot_auc[b] = auc_b

            # 每类准确率
            for cls in class_names:
                idxs_b = np.where(y_true_b == cls)[0]
                if idxs_b.size == 0:
                    boot_class_acc[cls][b] = np.nan
                else:
                    boot_class_acc[cls][b] = np.mean(y_pred_b[idxs_b] == cls)

        def ci_bounds(arr):
            arr_clean = arr[~np.isnan(arr)]
            lower = np.percentile(arr_clean, 2.5)
            upper = np.percentile(arr_clean, 97.5)
            return lower, upper

        acc_ci = ci_bounds(boot_acc)
        prec_ci = ci_bounds(boot_precision)
        rec_ci = ci_bounds(boot_recall)
        f1_ci = ci_bounds(boot_f1)
        auc_ci = ci_bounds(boot_auc)
        class_acc_ci = {cls: ci_bounds(boot_class_acc[cls]) for cls in class_names}

        print(f"\n[95% 置信区间]")
        print(f"▸ Overall Accuracy CI: [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]")
        for cls in class_names:
            lower, upper = class_acc_ci[cls]
            print(f"   · {cls} Accuracy CI: [{lower:.3f}, {upper:.3f}]")
        print(f"▸ Precision CI: [{prec_ci[0]:.3f}, {prec_ci[1]:.3f}]")
        print(f"▸ Recall    CI: [{rec_ci[0]:.3f}, {rec_ci[1]:.3f}]")
        print(f"▸ F1 Score  CI: [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]")
        print(f"▸ AUC       CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")

        # 10.3 绘制多分类 ROC 曲线
#         plt.figure(figsize=(7, 7))
#         colors = ["darkorange", "forestgreen", "royalblue"]
#         for i, cls in enumerate(class_names):
#             fpr, tpr, _ = roc_curve(true_bin[:, i], pred_probs[:, i])
#             auc_i = roc_auc_score(true_bin[:, i], pred_probs[:, i])
#             plt.plot(
#                 fpr, tpr,
#                 color=colors[i],
#                 lw=2,
#                 label=f"ROC {cls} (AUC = {auc_i:.3f})"
#             )# === 添加整体 ROC 曲线（基于是否预测正确）===
# binary_true = (true_labels == pred_labels).astype(int)  # 正确为1，错误为0
# correct_probs = np.max(pred_probs, axis=1)  # 最大置信度表示“被预测成某类的信心”

# fpr_all, tpr_all, _ = roc_curve(binary_true, correct_probs)
# auc_all = roc_auc_score(binary_true, correct_probs)

# plt.plot(
#     fpr_all, tpr_all,
#     color="black", linestyle="--", linewidth=2.2,
#     label=f"Overall ROC (Correct vs. Incorrect, AUC = {auc_all:.3f})"
# )
 
# plt.figure(figsize=(7, 7))
# colors = ["darkorange", "forestgreen", "royalblue"]
# for i, cls in enumerate(class_names):
#     fpr, tpr, _ = roc_curve(true_bin[:, i], pred_probs[:, i])
#     auc_i = roc_auc_score(true_bin[:, i], pred_probs[:, i])
#     plt.plot(
#         fpr, tpr,
#         color=colors[i],
#         lw=2,
#         label=f"ROC {cls} (AUC = {auc_i:.3f})"
# )
    
#         plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
#         plt.xlabel("False Positive Rate")
#         plt.ylabel("True Positive Rate")
#         plt.title("Multi‐Class ROC (One‐vs‐Rest)")
#         plt.legend(loc="lower right")
#         plt.grid(alpha=0.3)
#         plt.show()

# # 10.3 绘制多分类 ROC 曲线（每类 + 整体）
#         plt.figure(figsize=(7, 7))
#         colors = ["darkorange", "forestgreen", "royalblue"]
#         for i, cls in enumerate(class_names):
#             fpr, tpr, _ = roc_curve(true_bin[:, i], pred_probs[:, i])
#             auc_i = roc_auc_score(true_bin[:, i], pred_probs[:, i])
#             plt.plot(
#                 fpr, tpr,
#                 color=colors[i],
#                 lw=2,
#                 label=f"ROC {cls} (AUC = {auc_i:.3f})"
#     )

# # === 添加整体 ROC 曲线（正确 vs 错误）===
# # 预测正确为1，错误为0；用最大 softmax 置信度区分正确和错误
#         binary_true = (true_labels == pred_labels).astype(int)  # shape = (N,)
#         correct_probs = np.max(pred_probs, axis=1)              # shape = (N,)
#         fpr_all, tpr_all, _ = roc_curve(binary_true, correct_probs)
#         auc_all = roc_auc_score(binary_true, correct_probs)

#         plt.plot(
#             fpr_all, tpr_all,
#             color="black", linestyle="--", linewidth=2.2,
#             label=f"Overall ROC (Correct vs. Incorrect, AUC = {auc_all:.3f})"
#         )

#         # 参考线 & 格式
#         plt.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Random")
#         plt.xlabel("False Positive Rate")
#         plt.ylabel("True Positive Rate")
#         plt.title("Multi‐Class ROC (One‐vs‐Rest + Overall)")
#         plt.legend(loc="lower right")
#         plt.grid(alpha=0.3)
#         plt.tight_layout()
#         plt.show()


        # 10.3 绘制多分类 ROC 曲线（每类 + flatten ROC）

        plt.figure(figsize=(7, 7))

        # --- 每类的 One-vs-Rest ROC ---
        colors = ["darkorange", "forestgreen", "royalblue"]
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(true_bin[:, i], pred_probs[:, i])
            auc_i = roc_auc_score(true_bin[:, i], pred_probs[:, i])
            plt.plot(
                fpr, tpr,
                color=colors[i],
                lw=2,
                label=f"{cls} ROC (AUC = {auc_i:.3f})"
            )

        # --- Flatten ROC 曲线 ---
        y_true_flat = true_bin.ravel()          # (N * C,)
        y_score_flat = pred_probs.ravel()       # (N * C,)
        fpr_flat, tpr_flat, _ = roc_curve(y_true_flat, y_score_flat)
        auc_flat = roc_auc_score(y_true_flat, y_score_flat)
        plt.plot(
            fpr_flat, tpr_flat,
            color="black", linestyle="--", linewidth=2.5,
            label=f"Flatten ROC (All-class, AUC = {auc_flat:.3f})"
        )

        # --- 随机参考线 ---
        plt.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Random")

        # --- 图像格式 ---
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multi-Class ROC Curves (Per-Class + Flatten)")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()



        # 10.4 绘制混淆矩阵（带每行百分比）
        cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
        row_sums = cm.sum(axis=1, keepdims=True)  # 每行总数，形状 = (C,1)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_perc = np.divide(cm, row_sums, where=(row_sums != 0))

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                if row_sums[i, 0] == 0:
                    text = f"{count}\n(–)"
                else:
                    pct = cm_perc[i, j]
                    text = f"{count}\n({pct:.1%})"
                ax.text(
                    j, i, text,
                    ha="center", va="center",
                    color="white" if im.norm(count) > 0.5 else "black"
                )

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title("Confusion Matrix (Count & %)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    else:
        print("\n[警告] 标签数量不匹配，无法计算评价指标")

if __name__ == '__main__':
    main()





# # -*- coding: utf-8 -*-
# """
# Created on hzy
# Modified by ChatGPT to use Swin Transformer for prediction
# """

# import os
# import sys
# import json
# import tkinter as tk
# from tkinter import filedialog
# from collections import Counter

# import torch
# from PIL import Image
# import timm
# from torchvision import transforms
# from sklearn.metrics import (
#     confusion_matrix, ConfusionMatrixDisplay,
#     precision_score, recall_score, f1_score, roc_auc_score
# )
# from sklearn.preprocessing import label_binarize
# import matplotlib.pyplot as plt

# # 禁用科学计数法显示
# torch.set_printoptions(sci_mode=False)

# def load_model(model_path, num_classes, device):
#     """加载 Swin Transformer 模型"""
# # 注意要和 train_3class.py 里用的 model_name 一致
#     model = timm.create_model(
#         'swin_base_patch4_window7_224',
#         pretrained=False,
#         num_classes=num_classes,
#         img_size=(128, 128),
#         window_size=7
#     ).to(device)


#     state_dict = torch.load(model_path, map_location=device)
#     model.load_state_dict(state_dict)
#     model.eval()
#     return model

# def select_directory(title="选择文件夹"):
#     """GUI 目录选择对话框"""
#     root = tk.Tk()
#     root.withdraw()
#     path = filedialog.askdirectory(title=title)
#     return path if path else None

# def main():
#     # 设备配置
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f"\n[设备状态] 使用 {device} 进行推理")

#     # 交互式路径选择
#     print("\n[步骤1/3] 选择图像文件夹")
#     imgs_root = select_directory("选择包含类别子文件夹的图像根目录")
#     if not imgs_root:
#         print("错误：必须选择有效图像目录")
#         sys.exit()

#     print("\n[步骤2/3] 选择模型权重文件")
#     weights_path = filedialog.askopenfilename(
#         title="选择模型权重文件",
#         filetypes=[("PTH files", "*.pt"), ("PKL files", "*.pth")]
#     )
#     if not weights_path:
#         print("错误：必须选择有效权重文件")
#         sys.exit()

#     print("\n[步骤3/3] 选择类别索引文件")
#     json_path = filedialog.askopenfilename(
#         title="选择类别索引 JSON 文件",
#         filetypes=[("JSON files", "*.json")]
#     )
#     if not json_path:
#         print("错误：必须选择有效 JSON 文件")
#         sys.exit()

#     # 数据预处理 —— Swin Transformer 通常使用 224×224 输入
#     data_transform = transforms.Compose([
#         transforms.Resize(int(128*1.14)),
#         transforms.CenterCrop(128),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], 
#                              [0.229, 0.224, 0.225])
#     ])

#     # 收集所有图片路径和真实标签
#     img_paths, true_labels = [], []
#     for class_name in os.listdir(imgs_root):
#         class_dir = os.path.join(imgs_root, class_name)
#         if os.path.isdir(class_dir):
#             for img_name in os.listdir(class_dir):
#                 if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     img_paths.append(os.path.join(class_dir, img_name))
#                     true_labels.append(class_name)

#     # 加载类别索引
#     with open(json_path, 'r') as f:
#         class_indict = json.load(f)
#     class_names = [class_indict[str(i)] for i in range(len(class_indict))]

#     # 初始化模型
#     try:
#         model = load_model(weights_path, len(class_indict), device)
#         print("\n[模型信息]")
#         print(f"  · Backbone: Swin-base")
#         print(f"  · 类别数: {len(class_indict)}")
#         print(f"  · 权重路径: {weights_path}")
#     except Exception as e:
#         print(f"\n模型加载失败: {e}")
#         sys.exit()

#     # 批量预测
#     batch_size = 8
#     total = len(img_paths)
#     pred_labels = []

#     print("\n[预测进度]")
#     with torch.no_grad():
#         for i in range(0, total, batch_size):
#             batch_paths = img_paths[i:i+batch_size]
#             batch_imgs = []
#             for p in batch_paths:
#                 try:
#                     img = Image.open(p).convert("RGB")
#                     batch_imgs.append(data_transform(img))
#                 except Exception as e:
#                     print(f"跳过损坏图像 {p}: {e}")
#             if not batch_imgs:
#                 continue

#             inputs = torch.stack(batch_imgs).to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             batch_preds = [class_indict[str(idx.item())] for idx in preds]
#             pred_labels.extend(batch_preds)

#             # 进度条
#             prog = (i + len(batch_imgs)) / total
#             bar = '■' * int(prog * 30) + '□' * (30 - int(prog * 30))
#             print(f"\r进度: {bar} {prog*100:.1f}%", end="")

#     # 统计结果
#     print("\n\n[统计结果]")
#     cnt = Counter(pred_labels)
#     for cls, c in cnt.most_common():
#         print(f"▸ {cls:<15}: {c:>4} 张 ({c/len(pred_labels):.1%})")

#     # 评价指标 & 可视化
#     if len(true_labels) == len(pred_labels):
#         acc = sum(t==p for t,p in zip(true_labels, pred_labels)) / len(true_labels)
#         print(f"\n[整体准确率] {acc:.2%}")

#         print("\n[类别准确率]")
#         for cls in class_names:
#             idxs = [i for i,t in enumerate(true_labels) if t==cls]
#             if not idxs: continue
#             corr = sum(1 for i in idxs if pred_labels[i]==cls)
#             print(f"▸ {cls:<15}: {corr}/{len(idxs)} ({corr/len(idxs):.0%})")

#         # 混淆矩阵
#         cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
#         disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
#         disp.plot(cmap=plt.cm.Blues)
#         plt.title("Confusion Matrix")
#         plt.show()

#         # Precision / Recall / F1
#         prec = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
#         rec  = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
#         f1   = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
#         print(f"\n[评价指标]")
#         print(f"▸ Precision: {prec:.4f}")
#         print(f"▸ Recall:    {rec:.4f}")
#         print(f"▸ F1 Score:  {f1:.4f}")

#         # AUC (One-vs-Rest)
#         true_bin = label_binarize(true_labels, classes=class_names)
#         pred_bin = label_binarize(pred_labels, classes=class_names)
#         if true_bin.shape[1] == 1:
#             auc = roc_auc_score(true_bin, pred_bin)
#         else:
#             auc = roc_auc_score(true_bin, pred_bin, average='weighted', multi_class='ovr')
#         print(f"▸ AUC Score: {auc:.4f}")
        
        
        
#         # === 保存错分为 xianwei 的图像文件名 ===
#         misclassified_xianwei = []
#         for path, true, pred in zip(img_paths, true_labels, pred_labels):
#             if true == "xianwei" and pred != "xianwei":
#                 misclassified_xianwei.append(path)

#         output_txt = os.path.join(os.getcwd(), "xianwei_misclassified.txt")
#         with open(output_txt, "w", encoding="utf-8") as f:
#             for item in misclassified_xianwei:
#                 f.write(f"{item}\n")

#         print(f"\n[输出结果] 错分为非 dongmai 的图像数量: {len(misclassified_xianwei)}")
#         print(f"相关图像路径已保存至: {output_txt}")
        
        
        
#     else:
#         print("\n[警告] 标签数量不匹配，无法计算评价指标")

# if __name__ == '__main__':
#     main()

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from transformers import AutoImageProcessor  # RETFound官方推荐的图像预处理工具
import csv
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pycm import ConfusionMatrix
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from util.pos_embed import interpolate_pos_embed
import models_vit as models


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train == 'train':
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

class BilateralRetinaDataset(Dataset):
    def __init__(self, root_dir, split="train", num_classes=2, transform=None):
        self.root = os.path.join(root_dir, split, "bilateral")
        self.num_classes = num_classes
        self.transform = transform
        # 核心修改1：标签映射 → 抑郁=0，非抑郁=1
        self.class_map = {"class_depression": 0, "class_non_depression": 1}
        self.pair_list = self._collect_bilateral_pairs()
        
    def _collect_bilateral_pairs(self):
        pair_list = []
        for cls_name, cls_label in self.class_map.items():
            cls_dir = os.path.join(self.root, cls_name)
            if not os.path.exists(cls_dir):
                continue
            left_imgs = glob.glob(os.path.join(cls_dir, "*_left.jpg"))
            for left_path in left_imgs:
                right_path = left_path.replace("_left.jpg", "_right.jpg")
                if os.path.exists(right_path):
                    pair_list.append((left_path, right_path, cls_label))
        print(f"[{self.root}] 收集到 {len(pair_list)} 个双眼图像对")
        return pair_list

    def __len__(self):
        return len(self.pair_list)
    
    def __getitem__(self, idx):
        left_path, right_path, label = self.pair_list[idx]
        left_img = Image.open(left_path).convert("RGB")
        right_img = Image.open(right_path).convert("RGB")
        
        if self.transform is not None:
            left_processed = self.transform(left_img)
            right_processed = self.transform(right_img)
        
        return {
            "left_img": left_processed,
            "right_img": right_processed,
            "label": torch.tensor(label, dtype=torch.long)
        }

def build_bilateral_dataloader(args, split="train"):
    transform = build_transform(
        is_train=split,
        args=args
    )
    
    dataset = BilateralRetinaDataset(
        root_dir=args.data_path,
        split=split,
        num_classes=args.nb_classes,
        transform=transform
    )
    
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
        pin_memory=True
    )


# In[3]:


import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

def build_model(args):
    model_type = args.model
    pretrained_path = args.finetune

    if model_type == "RETFound_mae_bilateral":
        config = AutoConfig.from_pretrained(args.config_path)
        base_model = AutoModel.from_config(config)  
        hidden_dim = config.hidden_size
        feature_dim = config.hidden_size
        base_model.head = torch.nn.Linear(hidden_dim, args.nb_classes)
    
        if os.path.isfile(pretrained_path):
            print(f"加载本地预训练权重: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            checkpoint_model = checkpoint.get('model', checkpoint)

            base_model.pos_embed = base_model.embeddings.position_embeddings
            base_model.patch_embed = base_model.embeddings.patch_embeddings
            base_model.embeddings.num_position_embeddings = base_model.embeddings.position_embeddings.shape[1]

            checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
            checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
            checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}

            state_dict = base_model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    del checkpoint_model[k]

            interpolate_pos_embed(base_model, checkpoint_model)

            msg = base_model.load_state_dict(checkpoint_model, strict=False)
            print(f"权重加载完成，未匹配键: {msg.missing_keys[:5]}...")

            from timm.models.layers import trunc_normal_
            trunc_normal_(base_model.head.weight, std=2e-5)

        else:
            raise FileNotFoundError(f"预训练权重文件不存在: {pretrained_path}")

    class BilateralRETFoundModel(nn.Module):
        def __init__(self, base_model, feature_dim, num_classes, drop_path=0.1):
            super().__init__()
            self.base_model = base_model
            self.num_classes = num_classes

            self.fusion = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_path)
            )

            self.class_head = nn.Linear(feature_dim, num_classes)
            self._freeze_backbone_layers(freeze_ratio=0.9)

        def _freeze_backbone_layers(self, freeze_ratio=0.9):
            num_layers = len(self.base_model.encoder.layer)
            freeze_layers = int(num_layers * freeze_ratio)
            for i, layer in enumerate(self.base_model.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(f"冻结底层 {freeze_layers}/{num_layers} 个Transformer层")

        def extract_single_feat(self, img_tensor):
            outputs = self.base_model(pixel_values=img_tensor)
            return outputs.last_hidden_state[:, 0, :]

        def forward(self, batch):
            left_imgs = batch["left_img"]
            right_imgs = batch["right_img"]
            labels = batch["label"]

            left_feat = self.extract_single_feat(left_imgs)
            right_feat = self.extract_single_feat(right_imgs)

            fused_feat = torch.cat([left_feat, right_feat], dim=1)
            fused_feat = self.fusion(fused_feat)
            logits = self.class_head(fused_feat)

            return logits, labels

    model = BilateralRETFoundModel(
        base_model=base_model,
        feature_dim=feature_dim,
        num_classes=args.nb_classes,
        drop_path=args.drop_path
    )
    print(f"初始化双眼RETFound模型，分类头维度: {feature_dim}→{args.nb_classes}")

    return model


# In[ ]:


# ---------------------------
# 1. 验证函数（核心修改：类名改为英文，标签对应0=depression，1=non_depression）
# ---------------------------
def validate_one_epoch(args, model, val_loader, criterion, device):
    # 核心修改2：类名列表 → 英文，顺序对应标签0=depression，1=non_depression
    class_names = ["depression", "non_depression"]
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, labels = model(batch)
            
            # 注意：probs取索引1（non_depression）的概率，因roc_auc_score默认正类为1
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_avg_loss = total_loss / total_samples
    val_avg_acc = total_correct / total_samples

    # 分类报告：基于英文类名生成
    report = classification_report(
        all_labels, all_preds, target_names=class_names,
        output_dict=True, zero_division=0
    )
    val_macro_f1 = round(report["macro avg"]["f1-score"], 4)

    # 计算AUROC（正类为1=non_depression，与标签一致）
    try:
        val_auc = round(roc_auc_score(all_labels, all_probs), 4)
    except ValueError:
        val_auc = 0.0

    # 打印验证结果（英文类名）
    print(f"\n=== 验证集结果（Epoch {args.current_epoch+1}）===")
    print(f"Validation Loss: {val_avg_loss:.4f} | Accuracy: {val_avg_acc:.4f} | Macro F1: {val_macro_f1:.4f} | AUROC: {val_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    ))

    # 核心修改3：指标字典键名改为英文（depression/non_depression）
    val_metrics = {
        "epoch": args.current_epoch + 1,
        "val_avg_loss": round(val_avg_loss, 4),
        "val_avg_acc": round(val_avg_acc, 4),
        "val_macro_f1": val_macro_f1,
        "val_auc": val_auc,
        # 非抑郁：non_depression（标签1）
        "val_precision_non_depression": round(report["non_depression"]["precision"], 4),
        "val_recall_non_depression": round(report["non_depression"]["recall"], 4),
        "val_f1_non_depression": round(report["non_depression"]["f1-score"], 4),
        # 抑郁：depression（标签0）
        "val_precision_depression": round(report["depression"]["precision"], 4),
        "val_recall_depression": round(report["depression"]["recall"], 4),
        "val_f1_depression": round(report["depression"]["f1-score"], 4)
    }

    return val_avg_loss, val_avg_acc, val_macro_f1, val_metrics


# ---------------------------
# 2. 训练函数（同步修改类名和指标键名）
# ---------------------------
def train_one_epoch(args, model, dataloader, criterion, optimizer, device):
    class_names = ["depression", "non_depression"]
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for batch_idx, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, labels = model(batch)
        
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            print(f"Train Batch [{batch_idx+1}/{len(dataloader)}] | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

    train_avg_loss = total_loss / total_samples
    train_avg_acc = total_correct / total_samples
    
    report = classification_report(
        all_labels, all_preds, target_names=class_names,
        output_dict=True, zero_division=0
    )
    
    # 核心修改4：训练指标键名改为英文
    train_metrics = {
        "epoch": args.current_epoch + 1,
        "train_avg_loss": round(train_avg_loss, 4),
        "train_avg_acc": round(train_avg_acc, 4),
        # 非抑郁：non_depression（标签1）
        "train_precision_non_depression": round(report["non_depression"]["precision"], 4),
        "train_recall_non_depression": round(report["non_depression"]["recall"], 4),
        "train_f1_non_depression": round(report["non_depression"]["f1-score"], 4),
        # 抑郁：depression（标签0）
        "train_precision_depression": round(report["depression"]["precision"], 4),
        "train_recall_depression": round(report["depression"]["recall"], 4),
        "train_f1_depression": round(report["depression"]["f1-score"], 4),
        "train_macro_f1": round(report["macro avg"]["f1-score"], 4)
    }

    print(f"\n训练集Epoch总结（Epoch {args.current_epoch+1}）:")
    print(f"Train Loss: {train_avg_loss:.4f} | Accuracy: {train_avg_acc:.4f} | Macro F1: {train_metrics['train_macro_f1']:.4f}")

    return train_avg_loss, train_avg_acc, train_metrics


# ---------------------------
# 3. 测试函数（同步修改类名、指标键名和混淆矩阵）
# ---------------------------
def test_model(args, model, test_loader, criterion, device):
    class_names = ["depression", "non_depression"]
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, labels = model(batch)
            
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_avg_loss = total_loss / total_samples
    test_avg_acc = total_correct / total_samples

    report = classification_report(
        all_labels, all_preds, target_names=class_names,
        output_dict=True, zero_division=0
    )

    try:
        test_auc = round(roc_auc_score(all_labels, all_probs), 4)
    except ValueError:
        test_auc = 0.0

    # 生成混淆矩阵（标签0=depression，1=non_depression，确保映射正确）
    cm = ConfusionMatrix(actual_vector=all_labels, predict_vector=all_preds)
    # 混淆矩阵图像标注改为英文
    plt.figure(figsize=(8, 6))
    cm.plot(
        cmap=plt.cm.Blues, 
        number_label=True, 
        normalized=True, 
        plot_lib="matplotlib",
        title="Confusion Matrix (Test Set)",  # 英文标题
        class_name=class_names  # 英文类名标注（[depression, non_depression]）
    )
    plt.savefig(os.path.join(args.save_dir, 'confusion_matrix_test.jpg'), dpi=600, bbox_inches='tight')
    plt.close()  # 关闭图像避免内存占用
    
    print("\n=== Test Set Final Results ===")
    print(f"Test Loss: {test_avg_loss:.4f} | Accuracy: {test_avg_acc:.4f} | Macro F1: {round(report['macro avg']['f1-score'],4):.4f} | AUROC: {test_auc:.4f}")
    print("\nTest Set Classification Report:")
    print(classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    ))
    print("\nConfusion Matrix:")
    print(cm)

    
    cm_path = os.path.join(args.save_dir, "confusion_matrix_test.jpg")
    print(f"Confusion Matrix saved to: {cm_path}")

    # 核心修改5：测试指标键名改为英文（与训练/验证保持一致）
    test_metrics = {
        "epoch": "TEST",
        "test_avg_loss": round(test_avg_loss, 4),
        "test_avg_acc": round(test_avg_acc, 4),
        "test_auc": test_auc,
        # 非抑郁：non_depression（标签1）
        "test_precision_non_depression": round(report["non_depression"]["precision"], 4),
        "test_recall_non_depression": round(report["non_depression"]["recall"], 4),
        "test_f1_non_depression": round(report["non_depression"]["f1-score"], 4),
        # 抑郁：depression（标签0）
        "test_precision_depression": round(report["depression"]["precision"], 4),
        "test_recall_depression": round(report["depression"]["recall"], 4),
        "test_f1_depression": round(report["depression"]["f1-score"], 4),
        "test_macro_f1": round(report["macro avg"]["f1-score"], 4),
        # 混淆矩阵指标（TN/FP/FN/TP对应标签0=depression为正类时的定义，需确认一致性）
        # 注：pycm的TN/FP/FN/TP默认以“第一个类（class_names[0]=depression）”为正类
        "test_cm_TN": cm.TN,  # 真实non_depression（1），预测non_depression（1）
        "test_cm_FP": cm.FP,  # 真实non_depression（1），预测depression（0）
        "test_cm_FN": cm.FN,  # 真实depression（0），预测non_depression（1）
        "test_cm_TP": cm.TP   # 真实depression（0），预测depression（0）
    }

    return test_metrics, cm_path


# ---------------------------
# 4. 初始化与写入CSV（核心修改：表头全英文，适配新指标键名）
# ---------------------------
def init_metrics_csv(args):
    csv_path = os.path.join(args.save_dir, "metrics_log.csv")
    os.makedirs(args.save_dir, exist_ok=True)

    # 核心修改6：CSV表头全英文，顺序与指标字典对应
    csv_headers = [
        "epoch",
        # 训练指标（英文键名）
        "train_avg_loss", "train_avg_acc",
        "train_precision_non_depression", "train_recall_non_depression", "train_f1_non_depression",
        "train_precision_depression", "train_recall_depression", "train_f1_depression",
        "train_macro_f1",
        # 验证指标（英文键名）
        "val_avg_loss", "val_avg_acc", "val_macro_f1", "val_auc",
        "val_precision_non_depression", "val_recall_non_depression", "val_f1_non_depression",
        "val_precision_depression", "val_recall_depression", "val_f1_depression",
        # 测试指标（英文键名）
        "test_avg_loss", "test_avg_acc", "test_auc",
        "test_precision_non_depression", "test_recall_non_depression", "test_f1_non_depression",
        "test_precision_depression", "test_recall_depression", "test_f1_depression",
        "test_macro_f1",
        "test_cm_TN", "test_cm_FP", "test_cm_FN", "test_cm_TP"
    ]

    with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
    
    print(f"Metrics log initialized at: {csv_path}")
    return csv_path


def write_metrics_to_csv(metrics, csv_path):
    with open(csv_path, mode="a", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(open(csv_path, mode="r", encoding="utf-8"))
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # 确保缺失的指标（如测试阶段无训练指标）用空字符串填充
        row = {k: metrics.get(k, "") for k in fieldnames}
        writer.writerow(row)


# ---------------------------
# 5. 主函数（逻辑不变，仅因指标键名修改适配）
# ---------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.current_epoch = 0
    args.best_val_f1 = 0.0  # 仍以macro F1为最佳模型判断标准

    # 1. 构建数据加载器（标签映射已在Dataset中修改）
    print("\n=== Loading Datasets ===")
    train_loader = build_bilateral_dataloader(args, split="train")
    val_loader = build_bilateral_dataloader(args, split="val")
    test_loader = build_bilateral_dataloader(args, split="test")
    print(f"Train Loader: {len(train_loader)} batches | Val Loader: {len(val_loader)} batches | Test Loader: {len(test_loader)} batches")

    # 2. 构建模型（无修改，仅适配标签映射）
    print("\n=== Initializing Model ===")
    model = build_model(args).to(device)

    # 3. 初始化损失函数和优化器（若需加权损失，可在此处补充，如之前讨论的类别权重）
    # 若需解决类别不平衡，可替换为加权交叉熵：
    # criterion = nn.CrossEntropyLoss(weight=calculate_class_weights(args.data_path)).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 4. 初始化CSV日志（表头已改为英文）
    csv_path = init_metrics_csv(args)

    # 5. 训练循环（无逻辑修改，仅打印信息改为英文适配）
    print(f"\n=== Starting Training (Total Epochs: {args.epochs}) ===")
    for epoch in range(args.epochs):
        args.current_epoch = epoch
        
        # 训练（指标键名已改为英文）
        train_loss, train_acc, train_metrics = train_one_epoch(
            args, model, train_loader, criterion, optimizer, device
        )
        
        # 验证（指标键名已改为英文）
        val_loss, val_acc, val_macro_f1, val_metrics = validate_one_epoch(
            args, model, val_loader, criterion, device
        )
        
        # 写入指标（适配英文键名）
        combined_metrics = {**train_metrics, **val_metrics}
        write_metrics_to_csv(combined_metrics, csv_path)
        
        # 保存最佳模型（仍以验证集macro F1为准）
        if val_macro_f1 > args.best_val_f1:
            args.best_val_f1 = val_macro_f1
            best_model_path = os.path.join(args.save_dir, "best_bilateral_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"【Best Model Updated】Val Macro F1: {args.best_val_f1:.4f} | Saved to: {best_model_path}")
        else:
            print(f"【No Model Update】Current Val Macro F1: {val_macro_f1:.4f} < Best: {args.best_val_f1:.4f}")

    # 6. 测试过程（指标键名已改为英文）
    print("\n=== Starting Test (Using Best Model) ===")
    best_model_path = os.path.join(args.save_dir, "best_bilateral_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)
        test_metrics, cm_path = test_model(args, model, test_loader, criterion, device)
        
        # 写入测试结果（适配英文键名）
        write_metrics_to_csv(test_metrics, csv_path)
        print(f"Test metrics written to CSV: {csv_path}")
    else:
        raise FileNotFoundError(f"Best model not found at: {best_model_path}")

    # 最终总结（英文适配）
    print("\n=== Training & Testing Completed ===")
    print(f"Best Val Macro F1: {args.best_val_f1:.4f}")
    print(f"Full Metrics Log: {csv_path}")
    print(f"Confusion Matrix Image: {cm_path}")


# ---------------------------
# 6. 参数解析（无修改，保持原有功能）
# ---------------------------
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="RETFound_mae_bilateral", help="Model type")
    parser.add_argument("--finetune", type=str, required=True, help="Path to pretrained weights (.pth)")
    parser.add_argument("--nb_classes", type=int, default=2, help="Number of classes (2 for binary classification)")
    parser.add_argument("--data_path", type=str, required=True, help="Root directory of dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval for printing training logs")
    parser.add_argument("--save_dir", type=str, default="./output_dir/RETFound_mae_meh-Depression_two_eyes", help="Directory to save model and logs")
    # 预处理相关参数
    parser.add_argument("--input_size", type=int, default=224, help="Input image size (default: 224 for RETFound)")
    parser.add_argument("--color_jitter", type=float, default=0.4, help="Color jitter strength for training")
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1", help="Auto-augmentation strategy")
    parser.add_argument("--reprob", type=float, default=0.25, help="Random erase probability")
    parser.add_argument("--remode", type=str, default="pixel", help="Random erase mode (pixel/constant)")
    parser.add_argument("--recount", type=int, default=1, help="Number of random erase attempts")
    parser.add_argument("--config_path", type=str, required=True, help="Local path to RETFound model config.json")
    # Drop path parameter
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    return parser.parse_args()


# ---------------------------
# 入口函数（无修改）
# ---------------------------
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
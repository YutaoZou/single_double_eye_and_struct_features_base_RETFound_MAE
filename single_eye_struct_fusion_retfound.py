#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import glob
# import csv
# import numpy as np
# import matplotlib.pyplot as plt
# from pycm import ConfusionMatrix
# from sklearn.metrics import classification_report, roc_auc_score
# from timm.data import create_transform
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from transformers import AutoConfig, AutoModel
# import joblib

# # 导入必要的工具函数
# from util.pos_embed import interpolate_pos_embed
# import models_vit as models


# def build_transform(is_train, args):
#     mean = IMAGENET_DEFAULT_MEAN
#     std = IMAGENET_DEFAULT_STD
#     # 训练集变换
#     if is_train == 'train':
#         transform = create_transform(
#             input_size=args.input_size,
#             is_training=True,
#             color_jitter=args.color_jitter,
#             auto_augment=args.aa,
#             interpolation='bicubic',
#             re_prob=args.reprob,
#             re_mode=args.remode,
#             re_count=args.recount,
#             mean=mean,
#             std=std,
#         )
#         return transform

#     # 验证/测试集变换
#     t = []
#     if args.input_size <= 224:
#         crop_pct = 224 / 256
#     else:
#         crop_pct = 1.0
#     size = int(args.input_size / crop_pct)
#     t.append(
#         transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
#     )
#     t.append(transforms.CenterCrop(args.input_size))
#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(mean, std))
#     return transforms.Compose(t)


# class SingleEyeStructDataset(Dataset):
#     def __init__(self, root_dir, struct_feat_dir, split="train", num_classes=2, transform=None):
#         self.root = os.path.join(root_dir, split)
#         self.struct_feat_dir = struct_feat_dir
#         self.num_classes = num_classes
#         self.transform = transform
#         self.class_map = {"class_depression": 1, "class_non_depression": 0}
        
#         # 加载结构化特征
#         self.struct_feats = torch.load(
#             os.path.join(struct_feat_dir, f"{split}_struct_feats.pt")
#         )
        
#         # 收集单眼图像路径和对应索引
#         self.image_info = self._collect_single_eye_images()
        
#         # 验证结构化特征与图像数量匹配
#         assert len(self.image_info) == len(self.struct_feats), \
#             f"图像数量({len(self.image_info)})与结构化特征数量({len(self.struct_feats)})不匹配"

#     def _collect_single_eye_images(self):
#         image_info = []
#         for cls_name, cls_label in self.class_map.items():
#             cls_dir = os.path.join(self.root, cls_name)
#             if not os.path.exists(cls_dir):
#                 continue
#             # 收集所有单眼图像（假设为右眼图像）
#             eye_imgs = glob.glob(os.path.join(cls_dir, "*_*.*"))  # 修改为实际的单眼图像命名模式
#             for img_path in eye_imgs:
#                 # 提取样本ID用于匹配（根据实际文件名格式调整）
#                 img_name = os.path.basename(img_path)
#                 sample_id = img_name.split("_")[-4]  # 与之前提取ID的逻辑保持一致
#                 image_info.append({
#                     "path": img_path,
#                     "label": cls_label,
#                     "sample_id": sample_id
#                 })
#         print(f"[{self.root}] 收集到 {len(image_info)} 个单眼图像及对应结构化特征")
#         return image_info

#     def __len__(self):
#         return len(self.image_info)
    
#     def __getitem__(self, idx):
#         img_info = self.image_info[idx]
#         img = Image.open(img_info["path"]).convert("RGB")
#         label = img_info["label"]
#         struct_feat = self.struct_feats[idx]
        
#         if self.transform is not None:
#             img_processed = self.transform(img)
        
#         return {
#             "img": img_processed,
#             "struct_feat": struct_feat,
#             "label": torch.tensor(label, dtype=torch.long)
#         }


# def build_fusion_dataloader(args, split="train"):
#     transform = build_transform(
#         is_train=split,
#         args=args
#     )
    
#     dataset = SingleEyeStructDataset(
#         root_dir=args.data_path,
#         struct_feat_dir=args.struct_feat_path,
#         split=split,
#         num_classes=args.nb_classes,
#         transform=transform
#     )
    
#     return DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         shuffle=(split == "train"),
#         num_workers=args.num_workers,
#         pin_memory=True,
#         drop_last=(split == "train")
#     )


# def build_model(args):
#     model_type = args.model
#     pretrained_path = args.finetune

#     if model_type == "RETFound_fusion":
#         config = AutoConfig.from_pretrained(args.config_path)
#         base_model = AutoModel.from_config(config)  
#         hidden_dim = config.hidden_size
#         visual_feature_dim = config.hidden_size
        
#         # 加载预训练权重
#         if os.path.isfile(pretrained_path):
#             print(f"加载本地预训练权重: {pretrained_path}")
#             checkpoint = torch.load(pretrained_path, map_location='cpu')
#             checkpoint_model = checkpoint.get('model', checkpoint)

#             base_model.pos_embed = base_model.embeddings.position_embeddings
#             base_model.patch_embed = base_model.embeddings.patch_embeddings
#             base_model.embeddings.num_position_embeddings = base_model.embeddings.position_embeddings.shape[1]

#             checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
#             checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
#             checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}

#             state_dict = base_model.state_dict()
#             for k in ['head.weight', 'head.bias']:
#                 if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
#                     del checkpoint_model[k]

#             interpolate_pos_embed(base_model, checkpoint_model)
#             msg = base_model.load_state_dict(checkpoint_model, strict=False)
#             print(f"权重加载完成，未匹配键: {msg.missing_keys[:5]}...")

#             from timm.models.layers import trunc_normal_
#             # 为基础模型添加临时分类头用于权重初始化
#             base_model.head = nn.Linear(hidden_dim, args.nb_classes)
#             trunc_normal_(base_model.head.weight, std=2e-5)

#         else:
#             raise FileNotFoundError(f"预训练权重文件不存在: {pretrained_path}")

#     class FusionModel(nn.Module):
#         def __init__(self, base_model, visual_dim, struct_dim, num_classes, drop_path=0.1):
#             super().__init__()
#             self.base_model = base_model
#             self.num_classes = num_classes
            
#             # 结构化特征升维网络
#             self.struct_proj = nn.Sequential(
#                 nn.Linear(struct_dim, 256),
#                 nn.BatchNorm1d(256),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(drop_path),
#                 nn.Linear(256, visual_dim)
#             )
            
#             # 注意力融合模块
#             self.attention = nn.Sequential(
#                 nn.Linear(2 * visual_dim, visual_dim),
#                 nn.Tanh(),
#                 nn.Linear(visual_dim, 2),
#                 nn.Softmax(dim=1)
#             )
            
#             # 分类头
#             self.class_head = nn.Linear(visual_dim, num_classes)
#             self._freeze_backbone_layers(freeze_ratio=0.95)

#         def _freeze_backbone_layers(self, freeze_ratio=0.95):
#             num_layers = len(self.base_model.encoder.layer)
#             freeze_layers = int(num_layers * freeze_ratio)
#             for i, layer in enumerate(self.base_model.encoder.layer):
#                 if i < freeze_layers:
#                     for param in layer.parameters():
#                         param.requires_grad = False
#             print(f"冻结底层 {freeze_layers}/{num_layers} 个Transformer层")

#         def extract_visual_feat(self, img_tensor):
#             outputs = self.base_model(pixel_values=img_tensor)
#             return outputs.last_hidden_state[:, 0, :]  # CLS token

#         def forward(self, batch):
#             imgs = batch["img"]
#             struct_feats = batch["struct_feat"]
#             labels = batch["label"]

#             # 提取视觉特征
#             visual_feat = self.extract_visual_feat(imgs)  # [B, 1024]
            
#             # 结构化特征处理
#             struct_feat = self.struct_proj(struct_feats)  # [B, 1024]
            
#             # 注意力融合
#             concat_feat = torch.cat([visual_feat, struct_feat], dim=1)  # [B, 2048]
#             weights = self.attention(concat_feat)  # [B, 2]
#             fused_feat = visual_feat * weights[:, 0:1] + struct_feat * weights[:, 1:2]  # [B, 1024]
            
#             # 分类预测
#             logits = self.class_head(fused_feat)

#             return logits, labels

# #     # 加载结构化特征预处理管道以获取结构化特征维度
#     # 1. 添加辅助函数，用于检测实际的结构化特征维度
#     def check_struct_feature_dimension(struct_feat_path, split="train"):
#         """加载一个批次的结构化特征，检测实际维度"""
#         struct_feats = torch.load(os.path.join(struct_feat_path, f"{split}_struct_feats.pt"))
#         # 取第一个样本查看维度
#         if len(struct_feats) > 0:
#             actual_dim = struct_feats[0].shape[0]
#             print(f"检测到结构化特征实际维度: {actual_dim}")
#             return actual_dim
#         else:
#             raise ValueError("结构化特征文件为空，无法检测维度")
        
#     struct_dim = check_struct_feature_dimension(args.struct_feat_path)  # 实际的结构化特征维度值

#     model = FusionModel(
#         base_model=base_model,
#         visual_dim=visual_feature_dim,
#         struct_dim=struct_dim,
#         num_classes=args.nb_classes,
#         drop_path=args.drop_path
#     )
#     print(f"初始化单眼+结构化融合模型，结构化特征维度: {struct_dim}，视觉特征维度: {visual_feature_dim}")

#     return model


# def validate_one_epoch(args, model, val_loader, criterion, device):
#     class_names = ["depression", "non_depression"]
#     model.eval()
#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0
#     all_preds = []
#     all_labels = []
#     all_probs = []

#     with torch.no_grad():
#         for batch in val_loader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             logits, labels = model(batch)
            
#             probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
#             all_probs.extend(probs)
            
#             loss = criterion(logits, labels)
#             total_loss += loss.item() * labels.size(0)
            
#             preds = torch.argmax(logits, dim=1)
#             total_correct += (preds == labels).sum().item()
#             total_samples += labels.size(0)
            
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     val_avg_loss = total_loss / total_samples
#     val_avg_acc = total_correct / total_samples

#     report = classification_report(
#         all_labels, all_preds, target_names=class_names,
#         output_dict=True, zero_division=0
#     )
#     val_macro_f1 = round(report["macro avg"]["f1-score"], 4)

#     try:
#         val_auc = round(roc_auc_score(all_labels, all_probs), 4)
#     except ValueError:
#         val_auc = 0.0

#     print(f"\n=== 验证集结果（Epoch {args.current_epoch+1}）===")
#     print(f"Validation Loss: {val_avg_loss:.4f} | Accuracy: {val_avg_acc:.4f} | Macro F1: {val_macro_f1:.4f} | AUROC: {val_auc:.4f}")
#     print("\nClassification Report:")
#     print(classification_report(
#         all_labels, all_preds, target_names=class_names, zero_division=0
#     ))

#     val_metrics = {
#         "epoch": args.current_epoch + 1,
#         "val_avg_loss": round(val_avg_loss, 4),
#         "val_avg_acc": round(val_avg_acc, 4),
#         "val_macro_f1": val_macro_f1,
#         "val_auc": val_auc,
#         "val_precision_non_depression": round(report["non_depression"]["precision"], 4),
#         "val_recall_non_depression": round(report["non_depression"]["recall"], 4),
#         "val_f1_non_depression": round(report["non_depression"]["f1-score"], 4),
#         "val_precision_depression": round(report["depression"]["precision"], 4),
#         "val_recall_depression": round(report["depression"]["recall"], 4),
#         "val_f1_depression": round(report["depression"]["f1-score"], 4)
#     }

#     return val_avg_loss, val_avg_acc, val_macro_f1, val_metrics


# def train_one_epoch(args, model, dataloader, criterion, optimizer, scheduler, device):
#     class_names = ["depression", "non_depression"]
#     model.train()
#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0
#     all_preds = []
#     all_labels = []

#     for batch_idx, batch in enumerate(dataloader):
#         batch = {k: v.to(device) for k, v in batch.items()}
#         logits, labels = model(batch)
        
#         loss = criterion(logits, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item() * labels.size(0)
#         preds = torch.argmax(logits, dim=1)
#         total_correct += (preds == labels).sum().item()
#         total_samples += labels.size(0)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
        
#         if (batch_idx + 1) % args.log_interval == 0:
#             avg_loss = total_loss / total_samples
#             avg_acc = total_correct / total_samples
#             print(f"Train Batch [{batch_idx+1}/{len(dataloader)}] | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

#     train_avg_loss = total_loss / total_samples
#     train_avg_acc = total_correct / total_samples
#     scheduler.step() # 
#     report = classification_report(
#         all_labels, all_preds, target_names=class_names,
#         output_dict=True, zero_division=0
#     )
    
#     train_metrics = {
#         "epoch": args.current_epoch + 1,
#         "train_avg_loss": round(train_avg_loss, 4),
#         "train_avg_acc": round(train_avg_acc, 4),
#         "train_precision_non_depression": round(report["non_depression"]["precision"], 4),
#         "train_recall_non_depression": round(report["non_depression"]["recall"], 4),
#         "train_f1_non_depression": round(report["non_depression"]["f1-score"], 4),
#         "train_precision_depression": round(report["depression"]["precision"], 4),
#         "train_recall_depression": round(report["depression"]["recall"], 4),
#         "train_f1_depression": round(report["depression"]["f1-score"], 4),
#         "train_macro_f1": round(report["macro avg"]["f1-score"], 4)
#     }

#     print(f"\n训练集Epoch总结（Epoch {args.current_epoch+1}）:")
#     print(f"Train Loss: {train_avg_loss:.4f} | Accuracy: {train_avg_acc:.4f} | Macro F1: {train_metrics['train_macro_f1']:.4f}")

#     return train_avg_loss, train_avg_acc, train_metrics


# def test_model(args, model, test_loader, criterion, device):
#     class_names = ["depression", "non_depression"]
#     model.eval()
#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0
#     all_preds = []
#     all_labels = []
#     all_probs = []

#     with torch.no_grad():
#         for batch in test_loader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             logits, labels = model(batch)
            
#             probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
#             all_probs.extend(probs)
            
#             loss = criterion(logits, labels)
#             total_loss += loss.item() * labels.size(0)
            
#             preds = torch.argmax(logits, dim=1)
#             total_correct += (preds == labels).sum().item()
#             total_samples += labels.size(0)
            
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     test_avg_loss = total_loss / total_samples
#     test_avg_acc = total_correct / total_samples

#     report = classification_report(
#         all_labels, all_preds, target_names=class_names,
#         output_dict=True, zero_division=0
#     )

#     try:
#         test_auc = round(roc_auc_score(all_labels, all_probs), 4)
#     except ValueError:
#         test_auc = 0.0

#     # 生成混淆矩阵
#     cm = ConfusionMatrix(actual_vector=all_labels, predict_vector=all_preds)
#     plt.figure(figsize=(8, 6))
#     cm.plot(
#         cmap=plt.cm.Blues, 
#         number_label=True, 
#         normalized=True, 
#         plot_lib="matplotlib",
#         title="Confusion Matrix (Test Set)",
#         class_name=class_names
#     )
#     plt.savefig(os.path.join(args.save_dir, 'confusion_matrix_test.jpg'), dpi=600, bbox_inches='tight')
#     plt.close()
    
#     print("\n=== Test Set Final Results ===")
#     print(f"Test Loss: {test_avg_loss:.4f} | Accuracy: {test_avg_acc:.4f} | Macro F1: {round(report['macro avg']['f1-score'],4):.4f} | AUROC: {test_auc:.4f}")
#     print("\nTest Set Classification Report:")
#     print(classification_report(
#         all_labels, all_preds, target_names=class_names, zero_division=0
#     ))
#     print("\nConfusion Matrix:")
#     print(cm)

    
#     cm_path = os.path.join(args.save_dir, "confusion_matrix_test.jpg")
#     print(f"Confusion Matrix saved to: {cm_path}")

#     test_metrics = {
#         "epoch": "TEST",
#         "test_avg_loss": round(test_avg_loss, 4),
#         "test_avg_acc": round(test_avg_acc, 4),
#         "test_auc": test_auc,
#         "test_precision_non_depression": round(report["non_depression"]["precision"], 4),
#         "test_recall_non_depression": round(report["non_depression"]["recall"], 4),
#         "test_f1_non_depression": round(report["non_depression"]["f1-score"], 4),
#         "test_precision_depression": round(report["depression"]["precision"], 4),
#         "test_recall_depression": round(report["depression"]["recall"], 4),
#         "test_f1_depression": round(report["depression"]["f1-score"], 4),
#         "test_macro_f1": round(report["macro avg"]["f1-score"], 4),
#         "test_cm_TN": cm.TN,
#         "test_cm_FP": cm.FP,
#         "test_cm_FN": cm.FN,
#         "test_cm_TP": cm.TP
#     }

#     return test_metrics, cm_path


# def init_metrics_csv(args):
#     csv_path = os.path.join(args.save_dir, "metrics_log.csv")
#     os.makedirs(args.save_dir, exist_ok=True)

#     csv_headers = [
#         "epoch",
#         # 训练指标
#         "train_avg_loss", "train_avg_acc",
#         "train_precision_non_depression", "train_recall_non_depression", "train_f1_non_depression",
#         "train_precision_depression", "train_recall_depression", "train_f1_depression",
#         "train_macro_f1",
#         # 验证指标
#         "val_avg_loss", "val_avg_acc", "val_macro_f1", "val_auc",
#         "val_precision_non_depression", "val_recall_non_depression", "val_f1_non_depression",
#         "val_precision_depression", "val_recall_depression", "val_f1_depression",
#         # 测试指标
#         "test_avg_loss", "test_avg_acc", "test_auc",
#         "test_precision_non_depression", "test_recall_non_depression", "test_f1_non_depression",
#         "test_precision_depression", "test_recall_depression", "test_f1_depression",
#         "test_macro_f1",
#         "test_cm_TN", "test_cm_FP", "test_cm_FN", "test_cm_TP"
#     ]

#     with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=csv_headers)
#         writer.writeheader()
    
#     print(f"Metrics log initialized at: {csv_path}")
#     return csv_path


# def write_metrics_to_csv(metrics, csv_path):
#     with open(csv_path, mode="a", encoding="utf-8", newline="") as f:
#         reader = csv.DictReader(open(csv_path, mode="r", encoding="utf-8"))
#         fieldnames = reader.fieldnames
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         row = {k: metrics.get(k, "") for k in fieldnames}
#         writer.writerow(row)


# def main(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     args.current_epoch = 0
#     args.best_val_f1 = 0.0

#     # 构建数据加载器
#     print("\n=== Loading Datasets ===")
#     train_loader = build_fusion_dataloader(args, split="train")
#     val_loader = build_fusion_dataloader(args, split="val")
#     test_loader = build_fusion_dataloader(args, split="test")
#     print(f"Train Loader: {len(train_loader)} batches | Val Loader: {len(val_loader)} batches | Test Loader: {len(test_loader)} batches")

#     # 构建模型
#     print("\n=== Initializing Model ===")
#     model = build_model(args).to(device)

#     # 初始化损失函数和优化器
#     criterion = nn.CrossEntropyLoss().to(device)
#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=args.lr,
#         weight_decay=args.weight_decay
#     )
#     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

#     # 初始化CSV日志
#     csv_path = init_metrics_csv(args)

#     # 训练循环
#     print(f"\n=== Starting Training (Total Epochs: {args.epochs}) ===")
#     for epoch in range(args.epochs):
#         args.current_epoch = epoch
        
#         # 训练
#         train_loss, train_acc, train_metrics = train_one_epoch(
#             args, model, train_loader, criterion, optimizer, scheduler, device
#         )
        
#         # 验证
#         val_loss, val_acc, val_macro_f1, val_metrics = validate_one_epoch(
#             args, model, val_loader, criterion, device
#         )
        
#         # 写入指标
#         combined_metrics = {** train_metrics, **val_metrics}
#         write_metrics_to_csv(combined_metrics, csv_path)
        
#         # 保存最佳模型
#         if val_macro_f1 > args.best_val_f1:
#             args.best_val_f1 = val_macro_f1
#             best_model_path = os.path.join(args.save_dir, "best_fusion_model.pth")
#             torch.save(model.state_dict(), best_model_path)
#             print(f"【Best Model Updated】Val Macro F1: {args.best_val_f1:.4f} | Saved to: {best_model_path}")
#         else:
#             print(f"【No Model Update】Current Val Macro F1: {val_macro_f1:.4f} < Best: {args.best_val_f1:.4f}")

#     # 测试过程
#     print("\n=== Starting Test (Using Best Model) ===")
#     best_model_path = os.path.join(args.save_dir, "best_fusion_model.pth")
#     if os.path.exists(best_model_path):
#         model.load_state_dict(torch.load(best_model_path, map_location=device))
#         model.to(device)
#         test_metrics, cm_path = test_model(args, model, test_loader, criterion, device)
        
#         # 写入测试结果
#         write_metrics_to_csv(test_metrics, csv_path)
#         print(f"Test metrics written to CSV: {csv_path}")
#     else:
#         raise FileNotFoundError(f"Best model not found at: {best_model_path}")

#     # 最终总结
#     print("\n=== Training & Testing Completed ===")
#     print(f"Best Val Macro F1: {args.best_val_f1:.4f}")
#     print(f"Full Metrics Log: {csv_path}")
#     print(f"Confusion Matrix Image: {cm_path}")


# def parse_args():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, default="RETFound_fusion", help="Model type")
#     parser.add_argument("--finetune", type=str, required=True, help="Path to pretrained weights (.pth)")
#     parser.add_argument("--nb_classes", type=int, default=2, help="Number of classes")
#     parser.add_argument("--data_path", type=str, required=True, help="Root directory of image dataset")
#     parser.add_argument("--struct_feat_path", type=str, required=True, help="Directory of structured features")
#     parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
#     parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
#     parser.add_argument("--epochs", type=int, default=30, help="Total training epochs")
#     parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
#     parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
#     parser.add_argument("--log_interval", type=int, default=10, help="Training log interval")
#     parser.add_argument("--save_dir", type=str, default="./output_dir/RETFound_fusion_single_eye_struct", help="Output directory")
#     # 图像预处理参数
#     parser.add_argument("--input_size", type=int, default=224, help="Input image size")
#     parser.add_argument("--color_jitter", type=float, default=0.4, help="Color jitter strength")
#     parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1", help="Auto-augmentation strategy")
#     parser.add_argument("--reprob", type=float, default=0.25, help="Random erase probability")
#     parser.add_argument("--remode", type=str, default="pixel", help="Random erase mode")
#     parser.add_argument("--recount", type=int, default=1, help="Random erase count")
#     parser.add_argument("--config_path", type=str, required=True, help="RETFound config path")
#     parser.add_argument('--drop_path', type=float, default=0.1, help='Drop path rate')
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     os.makedirs(args.save_dir, exist_ok=True)
#     main(args)

      
# =======适配麻涌眼底彩照1317_脱敏图片新数据图片加表格特征=======

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from pycm import ConfusionMatrix
from sklearn.metrics import classification_report, roc_auc_score
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers import AutoConfig, AutoModel
import joblib
import pandas as pd

# 导入必要的工具函数
from util.pos_embed import interpolate_pos_embed
import models_vit as models


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # 训练集变换：关闭重复增强（已提前做数据增强）
    if is_train == 'train':
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=0.0,  # 关闭颜色抖动（避免过度增强）
            auto_augment=None,  # 关闭自动增强（已提前增强）
            interpolation='bicubic',
            re_prob=0.0,  # 关闭随机擦除（已提前增强）
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # 验证/测试集变换（保持不变）
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


class SingleEyeStructDataset(Dataset):
    def __init__(self, root_dir, struct_feat_dir, split="train", num_classes=2, transform=None):
        self.root = os.path.join(root_dir, split)
        self.struct_feat_dir = struct_feat_dir
        self.num_classes = num_classes
        self.transform = transform
        self.class_map = {"class_depression": 1, "class_non_depression": 0}
        
        # 加载结构化特征和对应的ID列表
        self.struct_feats = torch.load(
            os.path.join(struct_feat_dir, f"{split}_struct_feats.pt")
        )
        self.struct_ids = pd.read_csv(
            os.path.join(struct_feat_dir, f"{split}_struct_ids.csv"),
            encoding="gbk"
        )["pat_ID"].tolist()  # 读取结构化特征对应的ID列表
        
        # 建立ID→结构化特征索引的映射（关键：适配同一ID多图像）
        self.id_to_struct_idx = {id_str: idx for idx, id_str in enumerate(self.struct_ids)}
        
        # 收集单眼图像路径和信息（适配增强图）
        self.image_info = self._collect_single_eye_images()
        
        # 验证：图像数量可以大于结构化特征数量（因增强图存在）
        print(f"[{split}] 结构化特征数量: {len(self.struct_feats)} | 图像数量（含增强）: {len(self.image_info)}")

    def _extract_raw_id(self, img_name):
        """辅助函数：从增强图/原始图文件名中提取原始ID（适配多种格式）"""
        parts = img_name.split("_")
        # 1. 优先匹配原始逻辑（machong/shengyi前缀的ID）
        for part in parts:
            if part.startswith(("machong", "shengyi")):
                return part
        # 2. 匹配"right"或"left"后面的纯数字ID（如1637686）
        for i, part in enumerate(parts):
            # 修正逻辑：判断当前部分是"right"或"left"，且下一部分是数字
            if part in ("right", "left") and i + 1 < len(parts) and parts[i+1].isdigit():
                return parts[i+1]
        # 3. 若都不匹配，返回空字符串
        return ""

    def _collect_single_eye_images(self):
        image_info = []
        for cls_name, cls_label in self.class_map.items():
            cls_dir = os.path.join(self.root, cls_name)
            if not os.path.exists(cls_dir):
                continue
            # 匹配所有图像（含增强图aug_xxx_和原始图right_）
            eye_imgs = glob.glob(os.path.join(cls_dir, "*_*.*"))
            # 按原始ID排序（确保同一ID的图像连续，提升训练稳定性）
            eye_imgs.sort(key=lambda x: self._extract_raw_id(os.path.basename(x)))
            
            for img_path in eye_imgs:
                img_name = os.path.basename(img_path)
                # 提取原始ID（适配增强图格式：aug_1_left_machong0922_L_001.jpg）
                sample_id = int(self._extract_raw_id(img_name))  # ukb数据集用这行，注释下一行
#                 sample_id = self._extract_raw_id(img_name)  # 麻涌和省医数据集用这行，注释上一行
                if not sample_id or sample_id not in self.id_to_struct_idx:
                    print(f"警告：ID格式异常或未在结构化特征中找到，跳过文件 {img_name}")
                    continue
                image_info.append({
                    "path": img_path,
                    "label": cls_label,
                    "sample_id": sample_id,
                    "struct_idx": self.id_to_struct_idx[sample_id]  # 记录对应的结构化特征索引
                })
        print(f"[{self.root}] 成功收集到 {len(image_info)} 个有效图像（含增强图）")
        return image_info

    def __len__(self):
        return len(self.image_info)
    
    def __getitem__(self, idx):
        img_info = self.image_info[idx]
        img = Image.open(img_info["path"]).convert("RGB")
        label = img_info["label"]
        # 按记录的索引获取结构化特征（同一ID的增强图复用同一特征）
        struct_feat = self.struct_feats[img_info["struct_idx"]]
        
        if self.transform is not None:
            img_processed = self.transform(img)
        
        return {
            "img": img_processed,
            "struct_feat": struct_feat,
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": img_info["sample_id"]  # 可选：返回ID用于调试
        }


def build_fusion_dataloader(args, split="train"):
    transform = build_transform(
        is_train=split,
        args=args
    )
    
    dataset = SingleEyeStructDataset(
        root_dir=args.data_path,
        struct_feat_dir=args.struct_feat_path,
        split=split,
        num_classes=args.nb_classes,
        transform=transform
    )
    
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=(split == "train")
    )


def build_model(args):
    model_type = args.model
    pretrained_path = args.finetune

    if model_type == "RETFound_fusion":
        config = AutoConfig.from_pretrained(args.config_path)
        base_model = AutoModel.from_config(config)  
        hidden_dim = config.hidden_size
        visual_feature_dim = config.hidden_size
        
        # 加载预训练权重
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
            # 为基础模型添加临时分类头用于权重初始化
            base_model.head = nn.Linear(hidden_dim, args.nb_classes)
            trunc_normal_(base_model.head.weight, std=2e-5)

        else:
            raise FileNotFoundError(f"预训练权重文件不存在: {pretrained_path}")

    class FusionModel(nn.Module):
        def __init__(self, base_model, visual_dim, struct_dim, num_classes, drop_path=0.1):
            super().__init__()
            self.base_model = base_model
            self.num_classes = num_classes
            
            # 结构化特征升维网络（保持不变）
            self.struct_proj = nn.Sequential(
                nn.Linear(struct_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_path),
                nn.Linear(256, visual_dim)
            )
            
            # 注意力融合模块（保持不变）
            self.attention = nn.Sequential(
                nn.Linear(2 * visual_dim, visual_dim),
                nn.Tanh(),
                nn.Linear(visual_dim, 2),
                nn.Softmax(dim=1)
            )
            
            # 分类头（保持不变）
            self.class_head = nn.Linear(visual_dim, num_classes)
            self._freeze_backbone_layers(freeze_ratio=0.95)

        def _freeze_backbone_layers(self, freeze_ratio=0.95):
            num_layers = len(self.base_model.encoder.layer)
            freeze_layers = int(num_layers * freeze_ratio)
            for i, layer in enumerate(self.base_model.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(f"冻结底层 {freeze_layers}/{num_layers} 个Transformer层")

        def extract_visual_feat(self, img_tensor):
            outputs = self.base_model(pixel_values=img_tensor)
            return outputs.last_hidden_state[:, 0, :]  # CLS token

        def forward(self, batch):
            imgs = batch["img"]
            struct_feats = batch["struct_feat"]
            labels = batch["label"]

            # 提取视觉特征
            visual_feat = self.extract_visual_feat(imgs)  # [B, 1024]
            
            # 结构化特征处理
            struct_feat = self.struct_proj(struct_feats)  # [B, 1024]
            
            # 注意力融合
            concat_feat = torch.cat([visual_feat, struct_feat], dim=1)  # [B, 2048]
            weights = self.attention(concat_feat)  # [B, 2]
            fused_feat = visual_feat * weights[:, 0:1] + struct_feat * weights[:, 1:2]  # [B, 1024]
            
            # 分类预测
            logits = self.class_head(fused_feat)

            return logits, labels

    # 加载结构化特征预处理管道以获取结构化特征维度
    def check_struct_feature_dimension(struct_feat_path, split="train"):
        """加载一个批次的结构化特征，检测实际维度"""
        struct_feats = torch.load(os.path.join(struct_feat_path, f"{split}_struct_feats.pt"))
        if len(struct_feats) > 0:
            actual_dim = struct_feats[0].shape[0]
            print(f"检测到结构化特征实际维度: {actual_dim}")
            return actual_dim
        else:
            raise ValueError("结构化特征文件为空，无法检测维度")
        
    struct_dim = check_struct_feature_dimension(args.struct_feat_path)

    model = FusionModel(
        base_model=base_model,
        visual_dim=visual_feature_dim,
        struct_dim=struct_dim,
        num_classes=args.nb_classes,
        drop_path=args.drop_path
    )
    print(f"初始化单眼+结构化融合模型，结构化特征维度: {struct_dim}，视觉特征维度: {visual_feature_dim}")

    return model


def validate_one_epoch(args, model, val_loader, criterion, device):
    class_names = ["non_depression", "depression"]
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "sample_id"}  # 排除ID，不参与计算
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

    val_avg_loss = total_loss / total_samples
    val_avg_acc = total_correct / total_samples

    report = classification_report(
        all_labels, all_preds, target_names=class_names,
        output_dict=True, zero_division=0
    )
    val_macro_f1 = round(report["macro avg"]["f1-score"], 4)

    try:
        val_auc = round(roc_auc_score(all_labels, all_probs), 4)
    except ValueError:
        val_auc = 0.0

    print(f"\n=== 验证集结果（Epoch {args.current_epoch+1}）===")
    print(f"Validation Loss: {val_avg_loss:.4f} | Accuracy: {val_avg_acc:.4f} | Macro F1: {val_macro_f1:.4f} | AUROC: {val_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    ))

    val_metrics = {
        "epoch": args.current_epoch + 1,
        "val_avg_loss": round(val_avg_loss, 4),
        "val_avg_acc": round(val_avg_acc, 4),
        "val_macro_f1": val_macro_f1,
        "val_auc": val_auc,
        "val_precision_non_depression": round(report["non_depression"]["precision"], 4),
        "val_recall_non_depression": round(report["non_depression"]["recall"], 4),
        "val_f1_non_depression": round(report["non_depression"]["f1-score"], 4),
        "val_precision_depression": round(report["depression"]["precision"], 4),
        "val_recall_depression": round(report["depression"]["recall"], 4),
        "val_f1_depression": round(report["depression"]["f1-score"], 4)
    }

    return val_avg_loss, val_avg_acc, val_macro_f1, val_metrics


def train_one_epoch(args, model, dataloader, criterion, optimizer, scheduler, device):
    class_names = ["non_depression", "depression"]
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for batch_idx, batch in enumerate(dataloader):
        # 排除sample_id，不参与梯度计算
        batch = {k: v.to(device) for k, v in batch.items() if k != "sample_id"}
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
    scheduler.step()
    report = classification_report(
        all_labels, all_preds, target_names=class_names,
        output_dict=True, zero_division=0
    )
    
    train_metrics = {
        "epoch": args.current_epoch + 1,
        "train_avg_loss": round(train_avg_loss, 4),
        "train_avg_acc": round(train_avg_acc, 4),
        "train_precision_non_depression": round(report["non_depression"]["precision"], 4),
        "train_recall_non_depression": round(report["non_depression"]["recall"], 4),
        "train_f1_non_depression": round(report["non_depression"]["f1-score"], 4),
        "train_precision_depression": round(report["depression"]["precision"], 4),
        "train_recall_depression": round(report["depression"]["recall"], 4),
        "train_f1_depression": round(report["depression"]["f1-score"], 4),
        "train_macro_f1": round(report["macro avg"]["f1-score"], 4)
    }

    print(f"\n训练集Epoch总结（Epoch {args.current_epoch+1}）:")
    print(f"Train Loss: {train_avg_loss:.4f} | Accuracy: {train_avg_acc:.4f} | Macro F1: {train_metrics['train_macro_f1']:.4f}")

    return train_avg_loss, train_avg_acc, train_metrics


def test_model(args, model, test_loader, criterion, device):
    class_names = ["non_depression", "depression"]
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "sample_id"}
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

    # 生成混淆矩阵
    cm = ConfusionMatrix(actual_vector=all_labels, predict_vector=all_preds)
    plt.figure(figsize=(8, 6))
    cm.plot(
        cmap=plt.cm.Blues, 
        number_label=True, 
        normalized=True, 
        plot_lib="matplotlib",
        title="Confusion Matrix (Test Set)",
        class_name=class_names
    )
    plt.savefig(os.path.join(args.save_dir, 'confusion_matrix_test.jpg'), dpi=600, bbox_inches='tight')
    plt.close()
    
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

    test_metrics = {
        "epoch": "TEST",
        "test_avg_loss": round(test_avg_loss, 4),
        "test_avg_acc": round(test_avg_acc, 4),
        "test_auc": test_auc,
        "test_precision_non_depression": round(report["non_depression"]["precision"], 4),
        "test_recall_non_depression": round(report["non_depression"]["recall"], 4),
        "test_f1_non_depression": round(report["non_depression"]["f1-score"], 4),
        "test_precision_depression": round(report["depression"]["precision"], 4),
        "test_recall_depression": round(report["depression"]["recall"], 4),
        "test_f1_depression": round(report["depression"]["f1-score"], 4),
        "test_macro_f1": round(report["macro avg"]["f1-score"], 4),
        "test_cm_TN": cm.TN,
        "test_cm_FP": cm.FP,
        "test_cm_FN": cm.FN,
        "test_cm_TP": cm.TP
    }

    return test_metrics, cm_path


def init_metrics_csv(args):
    csv_path = os.path.join(args.save_dir, "metrics_log.csv")
    os.makedirs(args.save_dir, exist_ok=True)

    csv_headers = [
        "epoch",
        # 训练指标
        "train_avg_loss", "train_avg_acc",
        "train_precision_non_depression", "train_recall_non_depression", "train_f1_non_depression",
        "train_precision_depression", "train_recall_depression", "train_f1_depression",
        "train_macro_f1",
        # 验证指标
        "val_avg_loss", "val_avg_acc", "val_macro_f1", "val_auc",
        "val_precision_non_depression", "val_recall_non_depression", "val_f1_non_depression",
        "val_precision_depression", "val_recall_depression", "val_f1_depression",
        # 测试指标
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
        row = {k: metrics.get(k, "") for k in fieldnames}
        writer.writerow(row)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.current_epoch = 0
    args.best_val_f1 = 0.0

    # 构建数据加载器
    print("\n=== Loading Datasets ===")
    train_loader = build_fusion_dataloader(args, split="train")
    val_loader = build_fusion_dataloader(args, split="val")
    test_loader = build_fusion_dataloader(args, split="test")
    print(f"Train Loader: {len(train_loader)} batches | Val Loader: {len(val_loader)} batches | Test Loader: {len(test_loader)} batches")

    # 构建模型
    print("\n=== Initializing Model ===")
    model = build_model(args).to(device)

    # 初始化损失函数和优化器（适配少数类增强后的类别分布）
    # 可选：若增强后仍有类别不平衡，可添加权重
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # 初始化CSV日志
    csv_path = init_metrics_csv(args)

    # 训练循环
    print(f"\n=== Starting Training (Total Epochs: {args.epochs}) ===")
    for epoch in range(args.epochs):
        args.current_epoch = epoch
        
        # 训练
        train_loss, train_acc, train_metrics = train_one_epoch(
            args, model, train_loader, criterion, optimizer, scheduler, device
        )
        
        # 验证
        val_loss, val_acc, val_macro_f1, val_metrics = validate_one_epoch(
            args, model, val_loader, criterion, device
        )
        
        # 写入指标
        combined_metrics = {** train_metrics, **val_metrics}
        write_metrics_to_csv(combined_metrics, csv_path)
        
        # 保存最佳模型
        if val_macro_f1 > args.best_val_f1:
            args.best_val_f1 = val_macro_f1
            best_model_path = os.path.join(args.save_dir, "best_fusion_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"【Best Model Updated】Val Macro F1: {args.best_val_f1:.4f} | Saved to: {best_model_path}")
        else:
            print(f"【No Model Update】Current Val Macro F1: {val_macro_f1:.4f} < Best: {args.best_val_f1:.4f}")

    # 测试过程
    print("\n=== Starting Test (Using Best Model) ===")
    best_model_path = os.path.join(args.save_dir, "best_fusion_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)
        test_metrics, cm_path = test_model(args, model, test_loader, criterion, device)
        
        # 写入测试结果
        write_metrics_to_csv(test_metrics, csv_path)
        print(f"Test metrics written to CSV: {csv_path}")
    else:
        raise FileNotFoundError(f"Best model not found at: {best_model_path}")

    # 最终总结
    print("\n=== Training & Testing Completed ===")
    print(f"Best Val Macro F1: {args.best_val_f1:.4f}")
    print(f"Full Metrics Log: {csv_path}")
    print(f"Confusion Matrix Image: {cm_path}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="RETFound_fusion", help="Model type")
    parser.add_argument("--finetune", type=str, required=True, help="Path to pretrained weights (.pth)")
    parser.add_argument("--nb_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--data_path", type=str, required=True, help="Root directory of image dataset")
    parser.add_argument("--struct_feat_path", type=str, required=True, help="Directory of structured features")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--log_interval", type=int, default=10, help="Training log interval")
    parser.add_argument("--save_dir", type=str, default="./output_dir/RETFound_fusion_single_eye_struct", help="Output directory")
    # 图像预处理参数
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    parser.add_argument("--color_jitter", type=float, default=0.4, help="Color jitter strength（已在代码中强制设为0）")
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1", help="Auto-augmentation strategy（已在代码中强制设为None）")
    parser.add_argument("--reprob", type=float, default=0.25, help="Random erase probability（已在代码中强制设为0）")
    parser.add_argument("--remode", type=str, default="pixel", help="Random erase mode")
    parser.add_argument("--recount", type=int, default=1, help="Random erase count")
    parser.add_argument("--config_path", type=str, required=True, help="RETFound config path")
    parser.add_argument('--drop_path', type=float, default=0.1, help='Drop path rate')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)


    
####  麻涌数据集单独眼图特征  ####    
    
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import glob
# import csv
# import numpy as np
# import matplotlib.pyplot as plt
# from pycm import ConfusionMatrix
# from sklearn.metrics import classification_report, roc_auc_score
# from timm.data import create_transform
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from transformers import AutoConfig, AutoModel
# import pandas as pd

# # 导入必要的工具函数
# from util.pos_embed import interpolate_pos_embed
# import models_vit as models


# def build_transform(is_train, args):
#     mean = IMAGENET_DEFAULT_MEAN
#     std = IMAGENET_DEFAULT_STD
#     # 训练集变换：关闭重复增强（已提前做数据增强）
#     if is_train == 'train':
#         transform = create_transform(
#             input_size=args.input_size,
#             is_training=True,
#             color_jitter=0.0,  # 关闭颜色抖动（避免过度增强）
#             auto_augment=None,  # 关闭自动增强（已提前增强）
#             interpolation='bicubic',
#             re_prob=0.0,  # 关闭随机擦除（已提前增强）
#             re_mode=args.remode,
#             re_count=args.recount,
#             mean=mean,
#             std=std,
#         )
#         return transform

#     # 验证/测试集变换（保持不变）
#     t = []
#     if args.input_size <= 224:
#         crop_pct = 224 / 256
#     else:
#         crop_pct = 1.0
#     size = int(args.input_size / crop_pct)
#     t.append(
#         transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
#     )
#     t.append(transforms.CenterCrop(args.input_size))
#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(mean, std))
#     return transforms.Compose(t)


# class SingleEyeDataset(Dataset):
#     def __init__(self, root_dir, split="train", num_classes=2, transform=None):
#         self.root = os.path.join(root_dir, split)
#         self.num_classes = num_classes
#         self.transform = transform
#         self.class_map = {"class_depression": 1, "class_non_depression": 0}
        
#         # 收集单眼图像路径和信息（适配增强图）
#         self.image_info = self._collect_single_eye_images()
        
#         print(f"[{split}] 图像数量（含增强）: {len(self.image_info)}")

#     def _extract_raw_id(self, img_name):
#         """辅助函数：从增强图/原始图文件名中提取原始ID"""
#         parts = img_name.split("_")
#         for part in parts:
#             if part.startswith(("machong","shengyi")):
#                 return part
#         return ""

#     def _collect_single_eye_images(self):
#         image_info = []
#         for cls_name, cls_label in self.class_map.items():
#             cls_dir = os.path.join(self.root, cls_name)
#             if not os.path.exists(cls_dir):
#                 continue
#             # 匹配所有图像（含增强图aug_xxx_和原始图right_）
#             eye_imgs = glob.glob(os.path.join(cls_dir, "*_*.*"))
#             # 按原始ID排序（确保同一ID的图像连续，提升训练稳定性）
#             eye_imgs.sort(key=lambda x: self._extract_raw_id(os.path.basename(x)))
            
#             for img_path in eye_imgs:
#                 img_name = os.path.basename(img_path)
#                 # 提取原始ID
#                 sample_id = self._extract_raw_id(img_name)
#                 if not sample_id:
#                     print(f"警告：ID格式异常，跳过文件 {img_name}")
#                     continue
#                 image_info.append({
#                     "path": img_path,
#                     "label": cls_label,
#                     "sample_id": sample_id
#                 })
#         print(f"[{self.root}] 成功收集到 {len(image_info)} 个有效图像（含增强图）")
#         return image_info

#     def __len__(self):
#         return len(self.image_info)
    
#     def __getitem__(self, idx):
#         img_info = self.image_info[idx]
#         img = Image.open(img_info["path"]).convert("RGB")
#         label = img_info["label"]
        
#         if self.transform is not None:
#             img_processed = self.transform(img)
        
#         return {
#             "img": img_processed,
#             "label": torch.tensor(label, dtype=torch.long),
#             "sample_id": img_info["sample_id"]  # 可选：返回ID用于调试
#         }


# def build_dataloader(args, split="train"):
#     transform = build_transform(
#         is_train=split,
#         args=args
#     )
    
#     dataset = SingleEyeDataset(
#         root_dir=args.data_path,
#         split=split,
#         num_classes=args.nb_classes,
#         transform=transform
#     )
    
#     return DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         shuffle=(split == "train"),
#         num_workers=args.num_workers,
#         pin_memory=True,
#         drop_last=(split == "train")
#     )


# def build_model(args):
#     model_type = args.model
#     pretrained_path = args.finetune

#     if model_type == "RETFound":
#         config = AutoConfig.from_pretrained(args.config_path)
#         base_model = AutoModel.from_config(config)  
#         hidden_dim = config.hidden_size
#         visual_feature_dim = config.hidden_size
        
#         # 加载预训练权重
#         if os.path.isfile(pretrained_path):
#             print(f"加载本地预训练权重: {pretrained_path}")
#             checkpoint = torch.load(pretrained_path, map_location='cpu')
#             checkpoint_model = checkpoint.get('model', checkpoint)

#             base_model.pos_embed = base_model.embeddings.position_embeddings
#             base_model.patch_embed = base_model.embeddings.patch_embeddings
#             base_model.embeddings.num_position_embeddings = base_model.embeddings.position_embeddings.shape[1]

#             checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
#             checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
#             checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}

#             state_dict = base_model.state_dict()
#             for k in ['head.weight', 'head.bias']:
#                 if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
#                     del checkpoint_model[k]

#             interpolate_pos_embed(base_model, checkpoint_model)
#             msg = base_model.load_state_dict(checkpoint_model, strict=False)
#             print(f"权重加载完成，未匹配键: {msg.missing_keys[:5]}...")

#             from timm.models.layers import trunc_normal_
#             # 为基础模型添加分类头
#             base_model.head = nn.Linear(hidden_dim, args.nb_classes)
#             trunc_normal_(base_model.head.weight, std=2e-5)

#         else:
#             raise FileNotFoundError(f"预训练权重文件不存在: {pretrained_path}")

#     class ImageModel(nn.Module):
#         def __init__(self, base_model, visual_dim, num_classes, drop_path=0.1):
#             super().__init__()
#             self.base_model = base_model
#             self.num_classes = num_classes
            
#             # 分类头
#             self.class_head = nn.Linear(visual_dim, num_classes)
#             self._freeze_backbone_layers(freeze_ratio=0.95)

#         def _freeze_backbone_layers(self, freeze_ratio=0.95):
#             num_layers = len(self.base_model.encoder.layer)
#             freeze_layers = int(num_layers * freeze_ratio)
#             for i, layer in enumerate(self.base_model.encoder.layer):
#                 if i < freeze_layers:
#                     for param in layer.parameters():
#                         param.requires_grad = False
#             print(f"冻结底层 {freeze_layers}/{num_layers} 个Transformer层")

#         def extract_visual_feat(self, img_tensor):
#             outputs = self.base_model(pixel_values=img_tensor)
#             return outputs.last_hidden_state[:, 0, :]  # CLS token

#         def forward(self, batch):
#             imgs = batch["img"]
#             labels = batch["label"]

#             # 提取视觉特征
#             visual_feat = self.extract_visual_feat(imgs)  # [B, 1024]
            
#             # 分类预测
#             logits = self.class_head(visual_feat)

#             return logits, labels

#     model = ImageModel(
#         base_model=base_model,
#         visual_dim=visual_feature_dim,
#         num_classes=args.nb_classes,
#         drop_path=args.drop_path
#     )
#     print(f"初始化单眼图像模型，视觉特征维度: {visual_feature_dim}")

#     return model


# def validate_one_epoch(args, model, val_loader, criterion, device):
#     class_names = ["non_depression", "depression"]
#     model.eval()
#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0
#     all_preds = []
#     all_labels = []
#     all_probs = []

#     with torch.no_grad():
#         for batch in val_loader:
#             batch = {k: v.to(device) for k, v in batch.items() if k != "sample_id"}
#             logits, labels = model(batch)
            
#             probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
#             all_probs.extend(probs)
            
#             loss = criterion(logits, labels)
#             total_loss += loss.item() * labels.size(0)
            
#             preds = torch.argmax(logits, dim=1)
#             total_correct += (preds == labels).sum().item()
#             total_samples += labels.size(0)
            
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     val_avg_loss = total_loss / total_samples
#     val_avg_acc = total_correct / total_samples

#     report = classification_report(
#         all_labels, all_preds, target_names=class_names,
#         output_dict=True, zero_division=0
#     )
#     val_macro_f1 = round(report["macro avg"]["f1-score"], 4)

#     try:
#         val_auc = round(roc_auc_score(all_labels, all_probs), 4)
#     except ValueError:
#         val_auc = 0.0

#     print(f"\n=== 验证集结果（Epoch {args.current_epoch+1}）===")
#     print(f"Validation Loss: {val_avg_loss:.4f} | Accuracy: {val_avg_acc:.4f} | Macro F1: {val_macro_f1:.4f} | AUROC: {val_auc:.4f}")
#     print("\nClassification Report:")
#     print(classification_report(
#         all_labels, all_preds, target_names=class_names, zero_division=0
#     ))

#     val_metrics = {
#         "epoch": args.current_epoch + 1,
#         "val_avg_loss": round(val_avg_loss, 4),
#         "val_avg_acc": round(val_avg_acc, 4),
#         "val_macro_f1": val_macro_f1,
#         "val_auc": val_auc,
#         "val_precision_non_depression": round(report["non_depression"]["precision"], 4),
#         "val_recall_non_depression": round(report["non_depression"]["recall"], 4),
#         "val_f1_non_depression": round(report["non_depression"]["f1-score"], 4),
#         "val_precision_depression": round(report["depression"]["precision"], 4),
#         "val_recall_depression": round(report["depression"]["recall"], 4),
#         "val_f1_depression": round(report["depression"]["f1-score"], 4)
#     }

#     return val_avg_loss, val_avg_acc, val_macro_f1, val_metrics


# def train_one_epoch(args, model, dataloader, criterion, optimizer, scheduler, device):
#     class_names = ["non_depression", "depression"]
#     model.train()
#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0
#     all_preds = []
#     all_labels = []

#     for batch_idx, batch in enumerate(dataloader):
#         # 排除sample_id，不参与梯度计算
#         batch = {k: v.to(device) for k, v in batch.items() if k != "sample_id"}
#         logits, labels = model(batch)
        
#         loss = criterion(logits, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item() * labels.size(0)
#         preds = torch.argmax(logits, dim=1)
#         total_correct += (preds == labels).sum().item()
#         total_samples += labels.size(0)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
        
#         if (batch_idx + 1) % args.log_interval == 0:
#             avg_loss = total_loss / total_samples
#             avg_acc = total_correct / total_samples
#             print(f"Train Batch [{batch_idx+1}/{len(dataloader)}] | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

#     train_avg_loss = total_loss / total_samples
#     train_avg_acc = total_correct / total_samples
#     scheduler.step()
#     report = classification_report(
#         all_labels, all_preds, target_names=class_names,
#         output_dict=True, zero_division=0
#     )    
    
    
#     train_metrics = {
#         "epoch": args.current_epoch + 1,
#         "train_avg_loss": round(train_avg_loss, 4),
#         "train_avg_acc": round(train_avg_acc, 4),
#         "train_precision_non_depression": round(report["non_depression"]["precision"], 4),
#         "train_recall_non_depression": round(report["non_depression"]["recall"], 4),
#         "train_f1_non_depression": round(report["non_depression"]["f1-score"], 4),
#         "train_precision_depression": round(report["depression"]["precision"], 4),
#         "train_recall_depression": round(report["depression"]["recall"], 4),
#         "train_f1_depression": round(report["depression"]["f1-score"], 4),
#         "train_macro_f1": round(report["macro avg"]["f1-score"], 4)
#     }

#     print(f"\n训练集Epoch总结（Epoch {args.current_epoch+1}）:")
#     print(f"Train Loss: {train_avg_loss:.4f} | Accuracy: {train_avg_acc:.4f} | Macro F1: {train_metrics['train_macro_f1']:.4f}")

#     return train_avg_loss, train_avg_acc, train_metrics


# def test_model(args, model, test_loader, criterion, device):
#     class_names = ["non_depression", "depression"]
#     model.eval()
#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0
#     all_preds = []
#     all_labels = []
#     all_probs = []

#     with torch.no_grad():
#         for batch in test_loader:
#             batch = {k: v.to(device) for k, v in batch.items() if k != "sample_id"}
#             logits, labels = model(batch)
            
#             probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
#             all_probs.extend(probs)
            
#             loss = criterion(logits, labels)
#             total_loss += loss.item() * labels.size(0)
            
#             preds = torch.argmax(logits, dim=1)
#             total_correct += (preds == labels).sum().item()
#             total_samples += labels.size(0)
            
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     test_avg_loss = total_loss / total_samples
#     test_avg_acc = total_correct / total_samples

#     report = classification_report(
#         all_labels, all_preds, target_names=class_names,
#         output_dict=True, zero_division=0
#     )

#     try:
#         test_auc = round(roc_auc_score(all_labels, all_probs), 4)
#     except ValueError:
#         test_auc = 0.0

#     # 生成混淆矩阵
#     cm = ConfusionMatrix(actual_vector=all_labels, predict_vector=all_preds)
#     plt.figure(figsize=(8, 6))
#     cm.plot(
#         cmap=plt.cm.Blues, 
#         number_label=True, 
#         normalized=True, 
#         plot_lib="matplotlib",
#         title="Confusion Matrix (Test Set)",
#         class_name=class_names
#     )
#     plt.savefig(os.path.join(args.save_dir, 'confusion_matrix_test.jpg'), dpi=600, bbox_inches='tight')
#     plt.close()
    
#     print("\n=== Test Set Final Results ===")
#     print(f"Test Loss: {test_avg_loss:.4f} | Accuracy: {test_avg_acc:.4f} | Macro F1: {round(report['macro avg']['f1-score'],4):.4f} | AUROC: {test_auc:.4f}")
#     print("\nTest Set Classification Report:")
#     print(classification_report(
#         all_labels, all_preds, target_names=class_names, zero_division=0
#     ))
#     print("\nConfusion Matrix:")
#     print(cm)

#     cm_path = os.path.join(args.save_dir, "confusion_matrix_test.jpg")
#     print(f"Confusion Matrix saved to: {cm_path}")

#     test_metrics = {
#         "epoch": "TEST",
#         "test_avg_loss": round(test_avg_loss, 4),
#         "test_avg_acc": round(test_avg_acc, 4),
#         "test_auc": test_auc,
#         "test_precision_non_depression": round(report["non_depression"]["precision"], 4),
#         "test_recall_non_depression": round(report["non_depression"]["recall"], 4),
#         "test_f1_non_depression": round(report["non_depression"]["f1-score"], 4),
#         "test_precision_depression": round(report["depression"]["precision"], 4),
#         "test_recall_depression": round(report["depression"]["recall"], 4),
#         "test_f1_depression": round(report["depression"]["f1-score"], 4),
#         "test_macro_f1": round(report["macro avg"]["f1-score"], 4),
#         "test_cm_TN": cm.TN,
#         "test_cm_FP": cm.FP,
#         "test_cm_FN": cm.FN,
#         "test_cm_TP": cm.TP
#     }

#     return test_metrics, cm_path


# def init_metrics_csv(args):
#     csv_path = os.path.join(args.save_dir, "metrics_log.csv")
#     os.makedirs(args.save_dir, exist_ok=True)

#     csv_headers = [
#         "epoch",
#         # 训练指标
#         "train_avg_loss", "train_avg_acc",
#         "train_precision_non_depression", "train_recall_non_depression", "train_f1_non_depression",
#         "train_precision_depression", "train_recall_depression", "train_f1_depression",
#         "train_macro_f1",
#         # 验证指标
#         "val_avg_loss", "val_avg_acc", "val_macro_f1", "val_auc",
#         "val_precision_non_depression", "val_recall_non_depression", "val_f1_non_depression",
#         "val_precision_depression", "val_recall_depression", "val_f1_depression",
#         # 测试指标
#         "test_avg_loss", "test_avg_acc", "test_auc",
#         "test_precision_non_depression", "test_recall_non_depression", "test_f1_non_depression",
#         "test_precision_depression", "test_recall_depression", "test_f1_depression",
#         "test_macro_f1",
#         "test_cm_TN", "test_cm_FP", "test_cm_FN", "test_cm_TP"
#     ]

#     with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=csv_headers)
#         writer.writeheader()
    
#     print(f"Metrics log initialized at: {csv_path}")
#     return csv_path


# def write_metrics_to_csv(metrics, csv_path):
#     with open(csv_path, mode="a", encoding="utf-8", newline="") as f:
#         reader = csv.DictReader(open(csv_path, mode="r", encoding="utf-8"))
#         fieldnames = reader.fieldnames
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         row = {k: metrics.get(k, "") for k in fieldnames}
#         writer.writerow(row)


# def main(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     args.current_epoch = 0
#     args.best_val_f1 = 0.0

#     # 构建数据加载器
#     print("\n=== Loading Datasets ===")
#     train_loader = build_dataloader(args, split="train")
#     val_loader = build_dataloader(args, split="val")
#     test_loader = build_dataloader(args, split="test")
#     print(f"Train Loader: {len(train_loader)} batches | Val Loader: {len(val_loader)} batches | Test Loader: {len(test_loader)} batches")

#     # 构建模型
#     print("\n=== Initializing Model ===")
#     model = build_model(args).to(device)

#     # 初始化损失函数和优化器
#     criterion = nn.CrossEntropyLoss().to(device)
#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=args.lr,
#         weight_decay=args.weight_decay
#     )
#     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

#     # 初始化CSV日志
#     csv_path = init_metrics_csv(args)

#     # 训练循环
#     print(f"\n=== Starting Training (Total Epochs: {args.epochs}) ===")
#     for epoch in range(args.epochs):
#         args.current_epoch = epoch
        
#         # 训练
#         train_loss, train_acc, train_metrics = train_one_epoch(
#             args, model, train_loader, criterion, optimizer, scheduler, device
#         )
        
#         # 验证
#         val_loss, val_acc, val_macro_f1, val_metrics = validate_one_epoch(
#             args, model, val_loader, criterion, device
#         )
        
#         # 写入指标
#         combined_metrics = {**train_metrics, **val_metrics}
#         write_metrics_to_csv(combined_metrics, csv_path)
        
#         # 保存最佳模型
#         if val_macro_f1 > args.best_val_f1:
#             args.best_val_f1 = val_macro_f1
#             best_model_path = os.path.join(args.save_dir, "best_image_model.pth")
#             torch.save(model.state_dict(), best_model_path)
#             print(f"【Best Model Updated】Val Macro F1: {args.best_val_f1:.4f} | Saved to: {best_model_path}")
#         else:
#             print(f"【No Model Update】Current Val Macro F1: {val_macro_f1:.4f} < Best: {args.best_val_f1:.4f}")

#     # 测试过程
#     print("\n=== Starting Test (Using Best Model) ===")
#     best_model_path = os.path.join(args.save_dir, "best_image_model.pth")
#     if os.path.exists(best_model_path):
#         model.load_state_dict(torch.load(best_model_path, map_location=device))
#         model.to(device)
#         test_metrics, cm_path = test_model(args, model, test_loader, criterion, device)
        
#         # 写入测试结果
#         write_metrics_to_csv(test_metrics, csv_path)
#         print(f"Test metrics written to CSV: {csv_path}")
#     else:
#         raise FileNotFoundError(f"Best model not found at: {best_model_path}")

#     # 最终总结
#     print("\n=== Training & Testing Completed ===")
#     print(f"Best Val Macro F1: {args.best_val_f1:.4f}")
#     print(f"Full Metrics Log: {csv_path}")
#     print(f"Confusion Matrix Image: {cm_path}")


# def parse_args():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, default="RETFound", help="Model type (only image input)")
#     parser.add_argument("--finetune", type=str, required=True, help="Path to pretrained weights (.pth)")
#     parser.add_argument("--nb_classes", type=int, default=2, help="Number of classes")
#     parser.add_argument("--data_path", type=str, required=True, help="Root directory of image dataset")
#     parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
#     parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
#     parser.add_argument("--epochs", type=int, default=30, help="Total training epochs")
#     parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
#     parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
#     parser.add_argument("--log_interval", type=int, default=10, help="Training log interval")
#     parser.add_argument("--save_dir", type=str, default="./output_dir/RETFound_single_eye", help="Output directory")
#     # 图像预处理参数
#     parser.add_argument("--input_size", type=int, default=224, help="Input image size")
#     parser.add_argument("--color_jitter", type=float, default=0.4, help="Color jitter strength（已在代码中强制设为0）")
#     parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1", help="Auto-augmentation strategy（已在代码中强制设为None）")
#     parser.add_argument("--reprob", type=float, default=0.25, help="Random erase probability（已在代码中强制设为0）")
#     parser.add_argument("--remode", type=str, default="pixel", help="Random erase mode")
#     parser.add_argument("--recount", type=int, default=1, help="Random erase count")
#     parser.add_argument("--config_path", type=str, required=True, help="RETFound config path")
#     parser.add_argument('--drop_path', type=float, default=0.1, help='Drop path rate')
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     os.makedirs(args.save_dir, exist_ok=True)
#     main(args)    
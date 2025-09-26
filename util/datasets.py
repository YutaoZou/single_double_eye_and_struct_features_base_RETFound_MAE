import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import re

class BilateralRetinaDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir: 输出目录（如"D:/LingYi/data_pre_depression_bilateral"）
            split: 数据集类型（"train"/"val"/"test"）
            transform: 图像预处理（需与RETFound输入要求一致）
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.bilateral_samples = self._collect_bilateral_samples()  # 收集双眼配对样本
        self.unilateral_samples = self._collect_unilateral_samples()  # 收集单眼样本
        self.all_samples = self.bilateral_samples + self.unilateral_samples  # 合并所有样本

    def _collect_bilateral_samples(self):
        """收集双眼配对样本：通过文件名匹配左眼-右眼图像"""
        bilateral_dir = os.path.join(self.root_dir, self.split, "bilateral")
        if not os.path.exists(bilateral_dir):
            return []
        
        # 按抑郁类别遍历（class_depression/class_non_depression）
        class_dirs = [d for d in os.listdir(bilateral_dir) if os.path.isdir(os.path.join(bilateral_dir, d))]
        bilateral_samples = []
        
        for class_dir in class_dirs:
            class_path = os.path.join(bilateral_dir, class_dir)
            # 提取所有左眼图像（文件名含"_left.jpg"）
            left_files = [f for f in os.listdir(class_path) if f.endswith("_left.jpg")]
            
            for left_file in left_files:
                # 生成对应右眼文件名（替换"_left.jpg"为"_right.jpg"）
                right_file = left_file.replace("_left.jpg", "_right.jpg")
                right_path = os.path.join(class_path, right_file)
                
                # 检查右眼图像是否存在（确保配对完整）
                if os.path.exists(right_path):
                    left_path = os.path.join(class_path, left_file)
                    # 标签：1=抑郁（class_depression），0=无抑郁（class_non_depression）
                    label = 1 if "class_depression" in class_dir else 0
                    bilateral_samples.append({
                        "type": "bilateral",  # 标记为双眼样本
                        "left_path": left_path,
                        "right_path": right_path,
                        "label": label
                    })
        return bilateral_samples

    def _collect_unilateral_samples(self):
        """收集单眼样本（左眼/右眼）"""
        unilateral_types = ["unilateral_left", "unilateral_right"]
        unilateral_samples = []
        
        for eye_type in unilateral_types:
            eye_dir = os.path.join(self.root_dir, self.split, eye_type)
            if not os.path.exists(eye_dir):
                continue
            
            class_dirs = [d for d in os.listdir(eye_dir) if os.path.isdir(os.path.join(eye_dir, d))]
            for class_dir in class_dirs:
                class_path = os.path.join(eye_dir, class_dir)
                img_files = [f for f in os.listdir(class_path) if f.endswith((".jpg", ".jpeg", ".png"))]
                
                for img_file in img_files:
                    img_path = os.path.join(class_path, img_file)
                    label = 1 if "class_depression" in class_dir else 0
                    unilateral_samples.append({
                        "type": eye_type,  # 标记为单眼样本（unilateral_left/unilateral_right）
                        "img_path": img_path,
                        "label": label
                    })
        return unilateral_samples

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        label = torch.tensor(sample["label"], dtype=torch.long)
        
        if sample["type"] == "bilateral":
            # 读取双眼图像并预处理
            left_img = Image.open(sample["left_path"]).convert("RGB")
            right_img = Image.open(sample["right_path"]).convert("RGB")
            
            if self.transform:
                left_img = self.transform(left_img)
                right_img = self.transform(right_img)
            
            # 返回双眼图像（left/right）、标签、样本类型
            return {
                "left_img": left_img,
                "right_img": right_img,
                "label": label,
                "sample_type": "bilateral"
            }
        else:
            # 读取单眼图像并预处理
            img = Image.open(sample["img_path"]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            
            # 返回单眼图像、标签、样本类型
            return {
                "img": img,
                "label": label,
                "sample_type": sample["type"]
            }


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset

# 在 main_finetune.py 中找到数据集构建函数
# def build_dataset(is_train, args):
#     transform = build_transform(is_train, args)  # 保持原图像预处理（Normalize等）
    
#     if args.bilateral_mode:  # 新增：双眼模式
#         root = os.path.join(args.data_path, "train" if is_train else "val")
#         # 仅使用双眼配对样本（bilateral目录），单眼样本可后续单独训练
#         dataset = BilateralRetinaDataset(
#             root_dir=args.data_path,
#             split="train" if is_train else "val",
#             transform=transform
#         )
#         print(f"Loaded bilateral dataset: {len(dataset)} samples (is_train={is_train})")
#     else:
#         # 原单眼数据加载逻辑（保留，兼容单眼场景）
#         dataset = datasets.ImageFolder(
#             os.path.join(args.data_path, "train" if is_train else "val"),
#             transform=transform
#         )
    
#     return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train == 'train':
        # this should always dispatch to transforms_imagenet_train
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


# 修改 util/datasets.py 中的数据集类，区分少数类/多数类增强
import random
from torchvision import transforms
from PIL import Image
import torch

class RetinaAugDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, split='train', input_size=224, is_train=True, oversample_minority=True):
        self.data_path = os.path.join(data_path, split)
        self.is_train = is_train
        self.input_size = input_size
        self.oversample_minority = oversample_minority  # 2. 保存参数
        
        # 1. 定义类别（匹配你的文件夹名）
        self.classes = ['class_depression', 'class_non_depression']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 2. 加载所有样本路径和标签
        self.samples = []
        for cls in self.classes:
            cls_path = os.path.join(self.data_path, cls)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                self.samples.append((img_path, self.class_to_idx[cls]))
        
        # 3. 定义增强策略：区分少数类（抑郁）和多数类（非抑郁）
        self._init_augmentations()        
        
        # 4. 处理过采样（仅在训练模式且 oversample_minority=True 时生效）
        if self.is_train and self.oversample_minority:
            # 分离少数类（抑郁）和多数类（非抑郁）的索引
            self.minority_indices = [idx for idx, (_, label) in enumerate(self.samples) if label == 0]
            self.majority_indices = [idx for idx, (_, label) in enumerate(self.samples) if label == 1]
            
            # 过采样目标：少数类样本数 = 多数类样本数 × 0.33（1:3 比例）
            self.target_minority_num = int(len(self.majority_indices) * 0.66)
            self.extra_minority_num = self.target_minority_num - len(self.minority_indices)
            print(f"过采样配置：少数类原始{len(self.minority_indices)}个，目标{self.target_minority_num}个（额外生成{self.extra_minority_num}个）")
            
            # 随机重复少数类索引（配合增强生成多样化样本）
            self.augmented_indices = self.majority_indices + self.minority_indices + \
                                     random.choices(self.minority_indices, k=max(0, self.extra_minority_num))
        else:
            # 不启用过采样时，直接使用原始样本索引
            self.augmented_indices = list(range(len(self.samples)))

    def _init_augmentations(self):
        # 1. 基础PIL预处理（仅Resize，转Tensor前的操作）
        self.pil_preprocess = transforms.Compose([
            transforms.Resize((self.input_size + 48, self.input_size + 48)),  # 放大
        ])

        # 2. 多数类（非抑郁）的PIL增强（仅对PIL Image操作）
        self.majority_pil_aug = transforms.Compose([
            transforms.RandomCrop((self.input_size, self.input_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])

        # 3. 少数类（抑郁）的PIL增强（仅对PIL Image操作）
        self.minority_pil_aug = transforms.Compose([
            transforms.RandomCrop((self.input_size, self.input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-8, 8)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ])

        # 4. Tensor增强（转Tensor后，对Tensor操作）
        self.tensor_aug = transforms.Compose([
            transforms.ToTensor(),  # 关键：先转为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # 对Tensor的增强（如高斯模糊、随机擦除）
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3))], p=0.4),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.08)),
        ])

        # 5. 验证集仅做PIL预处理+Tensor转换（无增强）
        self.val_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        real_idx = self.augmented_indices[idx]
        img_path, label = self.samples[real_idx]

        # 1. 打开图像（PIL Image格式）
        img = Image.open(img_path).convert('RGB')

        if self.is_train:
            # 2. 先应用PIL预处理（放大）
            img = self.pil_preprocess(img)

            # 3. 根据类别应用PIL增强（此时仍是PIL Image）
            if label == 0:  # 少数类（抑郁）
                img = self.minority_pil_aug(img)
            else:  # 多数类（非抑郁）
                img = self.majority_pil_aug(img)

            # 4. 转为Tensor并应用Tensor增强（关键：此时已转为Tensor，支持shape属性）
            img = self.tensor_aug(img)
        else:
            # 验证集：仅基础变换
            img = self.val_transform(img)

        return img, label

    def __len__(self):
        return len(self.augmented_indices)

    
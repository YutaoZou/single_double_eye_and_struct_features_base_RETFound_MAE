# 眼底图抑郁分类检测项目

本仓库以 **RetFound** ([https://github.com/rmaphoh/RETFound.git](https://github.com/rmaphoh/RETFound.git)) 为基础，专注于眼底图像的抑郁分类检测任务。

项目主要包含两大部分：
1.  **数据预处理**：对眼底图像和表格结构化特征进行预处理。
2.  **模型训练与评估**：提供了三种不同的抑郁分类检测模型，分别对应单眼、双眼和多模态融合的方法。

---

## 主要功能与文件

| 功能 (Function) | 对应脚本 (Script) | 说明 (Description) |
| :--- | :--- | :--- |
| **数据预处理** | `preprocess_eye_img.py` | 处理原始眼底图像和结构化数据，生成模型可直接使用的输入文件。 |
| **单眼抑郁分类** | `main_finetune.py` | 使用单眼眼底图像进行抑郁分类。 |
| **双眼抑郁分类** | `double_eyes_retfound.py` | 同时使用左右眼眼底图像进行抑郁分类。 |
| **多模态融合分类** | `single_eye_struct_fusion_retfound.py` | 融合单眼眼底图像和结构化特征进行抑郁分类。 |

---

## 快速开始

### 步骤 1: 数据预处理

首先，运行预处理脚本对原始数据进行处理。

```bash
python preprocess_eye_img.py
```

### 步骤 2: 模型训练与评估

1. 单眼
```bash
python main_finetune.py ^
     --model RETFound_mae ^
     --savemodel ^
     --global_pool ^
     --batch_size 8 ^
     --accum_iter 2 ^
     --num_workers 0 ^
     --epochs 50 ^
     --blr 5e-3 --layer_decay 0.65 ^
     --weight_decay 0.1 --drop_path 0.3 ^
     --nb_classes 2 ^
     --data_path "运行preprocess_eye_img后对应的眼图输出文件路径" ^
     --input_size 224 ^
     --task RETFound_mae_meh-Depression ^
     --finetune "RETFound_mae_natureCFP.pth预训练权重路径"
```

2. 双眼
```bash
python double_eyes_retfound.py ^
  --model RETFound_mae_bilateral ^
  --finetune "RETFound_mae_natureCFP.pth预训练权重路径" ^
  --data_path "运行preprocess_eye_img后对应的眼图输出文件路径" ^
  --batch_size 16 ^
  --epochs 50 ^
  --lr 1e-4 ^
  --save_dir ./output_dir/RETFound_mae_meh-Depression_two_eyes ^
  --config_path "RETFound_MAE的config.json路径"
```

3. 多模态
```bash
python single_eye_struct_fusion_retfound.py ^
  --model RETFound_fusion ^
  --finetune "RETFound_mae_natureCFP.pth预训练权重路径" ^
  --data_path "运行preprocess_eye_img后对应的眼图输出文件路径" ^
  --struct_feat_path "运行preprocess_eye_img后对应的结构化特征输出文件路径" ^
  --batch_size 16 ^
  --epochs 50 ^
  --lr 1e-4 ^
  --save_dir ./output_dir/RETFound_fusion_single_left_eye_struct ^
  --config_path "RETFound_MAE的config.json路径"
```

## 参考与致谢
1. 基础模型：RetFound（[GitHub 仓库]<https://github.com/rmaphoh/RETFound.git>）
2. 论文参考：Zhou, Y., et al. "RETFound: a foundation model for generalizable disease detection from retinal images." Nature (2023).

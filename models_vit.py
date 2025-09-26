# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1,keepdim=True)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def RETFound_mae(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def RETFound_dinov2(args, **kwargs):
    model = timm.create_model(
        'vit_large_patch14_dinov2.lvd142m',
        pretrained=True,
        img_size=224,
        **kwargs
    )
    return model



def build_model(args):
    model_type = args.model
    pretrained_path = args.finetune
    
    if model_type == "RETFound_mae_bilateral":
        # 加载RETFound MAE预训练模型
        config = AutoConfig.from_pretrained(pretrained_path)
        base_model = AutoModel.from_pretrained(pretrained_path, config=config)
        feature_dim = config.hidden_size
        
        # 定义双眼模型（含特征融合）
        class BilateralRETFoundModel(nn.Module):
            def __init__(self, base_model, feature_dim, num_classes):
                super().__init__()
                self.base_model = base_model
                # 双眼特征融合模块（参考论文自适应融合）
                self.fusion = nn.Sequential(
                    nn.Linear(feature_dim * 2, feature_dim),
                    nn.BatchNorm1d(feature_dim),
                    nn.ReLU(),
                    nn.Dropout(args.drop_path)
                )
                # 分类头（与原单眼模型一致）
                self.head = nn.Linear(feature_dim, num_classes)
            
            def extract_feat(self, img):
                # 提取单张图像特征（CLS token）
                outputs = self.base_model(pixel_values=img)
                return outputs.last_hidden_state[:, 0, :]  # (batch_size, feature_dim)
            
            def forward(self, batch):
                # 从batch中分离左眼/右眼图像（需修改dataloader输出）
                left_imgs = batch["left_img"]
                right_imgs = batch["right_img"]
                labels = batch["label"]
                
                # 分别提取双眼特征
                left_feat = self.extract_feat(left_imgs)
                right_feat = self.extract_feat(right_imgs)
                
                # 融合特征并分类
                fused_feat = torch.cat([left_feat, right_feat], dim=1)
                fused_feat = self.fusion(fused_feat)
                logits = self.head(fused_feat)
                
                return logits, labels
        
        # 初始化双眼模型
        model = BilateralRETFoundModel(
            base_model=base_model,
            feature_dim=feature_dim,
            num_classes=args.nb_classes
        )
    
    else:
        # 原单眼模型逻辑（保留）
        model = create_retfound_mae_model(args)
    
    return model


import argparse
import datetime
import json

import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup

import models_vit as models
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from huggingface_hub import hf_hub_download, login
from engine_finetune import train_one_epoch, evaluate

import warnings
import faulthandler

from util.datasets import RetinaAugDataset

faulthandler.enable()
warnings.simplefilter(action='ignore', category=FutureWarning)

# 1. 定义Focal Loss类
import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, gamma=3.0, class_weights=None):
        super().__init__()
        self.alpha = alpha  # Focal Loss的全局平衡因子（默认1.0，无需修改）
        self.gamma = gamma  # 聚焦因子（3.0适合难分样本，已按你的需求保留）
        self.class_weights = class_weights  # 类别权重（[抑郁权重, 非抑郁权重]，维度[2]）

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型输出，shape=[batch_size, 2]（双类别logits，无softmax）
            targets: 标签，shape=[batch_size]（类别索引，0=抑郁，1=非抑郁）
        Returns:
            平均Focal Loss
        """
        # 1. 计算基础交叉熵损失（支持类别权重，适配双输出+类别索引）
        # 注意：用F.cross_entropy（函数式），而非类实例化，直接传入类别权重
        ce_loss = F.cross_entropy(
            input=inputs,          # [batch,2]：模型输出logits
            target=targets,        # [batch]：类别索引
            weight=self.class_weights,  # [2]：类别权重（抑郁→索引0，非抑郁→索引1）
            reduction='none'       # 不自动平均，保留每个样本的损失值（用于后续Focal计算）
        )  # ce_loss shape: [batch_size]

        # 2. 计算每个样本的“正确类别概率”pt（关键：双输出需用softmax）
        # 步骤：先对inputs做softmax得到类别概率 → 按targets索引取“正确类别”的概率
        softmax_probs = F.softmax(inputs, dim=1)  # [batch,2]：每个样本的两类概率和为1
        pt = softmax_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)  
        # 解释：
        # - targets.unsqueeze(1)：将[batch]→[batch,1]（匹配softmax_probs的dim=1）
        # - gather：按索引取“正确类别”的概率 → [batch,1]
        # - squeeze(1)：还原为[batch]，每个值是当前样本“正确类别”的概率
        # pt shape: [batch_size]

        # 3. 计算Focal Loss的核心项：(1-pt)^gamma（放大难分样本损失）
        focal_term = (1.0 - pt) ** self.gamma  # [batch_size]

        # 4. 计算最终Focal Loss（全局平衡因子alpha + 类别权重已在ce_loss中包含）
        focal_loss = self.alpha * focal_term * ce_loss  # [batch_size]

        # 5. 返回批次平均损失
        return focal_loss.mean()

def get_class_weights(data_path, split='train'):
    """
    针对文件夹名称：depression（少数类）、non-depression（多数类）
    手动固定类别顺序：[non-depression, depression]，确保权重计算正确
    """
    # 1. 手动指定类别顺序（多数类在前，少数类在后）
    target_class_order = ['class_non_depression', 'class_depression']
    # 2. 检查文件夹是否存在（避免路径错误）
    class_dirs = os.listdir(os.path.join(data_path, split))
    for cls in target_class_order:
        if cls not in class_dirs:
            raise FileNotFoundError(f"类别文件夹 {cls} 不存在于 {os.path.join(data_path, split)}，当前文件夹：{class_dirs}")
    # 3. 统计每个类别的样本数（按手动指定的顺序）
    class_counts = []
    for cls in target_class_order:
        cls_path = os.path.join(data_path, split, cls)
        sample_cnt = len(os.listdir(cls_path))
        class_counts.append(sample_cnt)
        print(f"类别 {cls}: {sample_cnt} 个样本")  # 打印样本数，确认多数/少数类
    # 4. 计算类别权重（总样本数 / (类别数 × 该类样本数)）
    total_samples = sum(class_counts)
    num_classes = len(target_class_order)
    # 降低少数类权重：乘以0.4（3.76×0.4≈1.5，与多数类0.58的比例约2.6:1）
    class_weights = []
    for i, cnt in enumerate(class_counts):
        base_weight = total_samples / (num_classes * cnt)
        if i == 1:  # 少数类（class_depression）
            class_weights.append(base_weight * 0.4)  # 核心：降低权重
        else:  # 多数类
            class_weights.append(base_weight)
    
    print(f"调整后权重（非抑郁:抑郁）: {class_weights}")  # 预期输出：[0.576, ~1.5]
    return class_weights, target_class_order, class_counts

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.65,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=15, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--task', default='', type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=8, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_logs',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # fine-tuning parameters
    parser.add_argument('--savemodel', action='store_true', default=True,
                        help='Save model')
    parser.add_argument('--norm', default='IMAGENET', type=str, help='Normalization method')
    parser.add_argument('--enhance', action='store_true', default=False, help='Use enhanced data')
    parser.add_argument('--datasets_seed', default=2026, type=int)

    return parser


def main(args, criterion):
    if args.resume and not args.eval:
        resume = args.resume
        checkpoint = torch.load(args.resume, map_location='cpu')
        print("Load checkpoint from: %s" % args.resume)
        args = checkpoint['args']
        args.resume = resume

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.model=='RETFound_mae':
        model = models.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    else:
        model = models.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            args=args,
        )
    
#     if args.finetune and not args.eval:
        
#         print(f"Downloading pre-trained weights from: {args.finetune}")
        
#         checkpoint_path = hf_hub_download(
#             repo_id=f'YukunZhou/{args.finetune}',
#             filename=f'{args.finetune}.pth',
#         )
        
    if args.finetune and not args.eval:
        # 判断是否为本地文件路径（存在且是文件）
        if os.path.isfile(args.finetune):
            print(f"Loading local pre-trained weights from: {args.finetune}")
            checkpoint_path = args.finetune  # 直接使用本地路径
        else:
            # 若不是本地文件，再从HuggingFace下载
            print(f"Downloading pre-trained weights from HuggingFace: {args.finetune}")
            checkpoint_path = hf_hub_download(
                repo_id=f'YukunZhou/{args.finetune}',
                filename=f'{args.finetune}.pth',
            )        
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        
        
        if args.model!='RETFound_mae':
            checkpoint_model = checkpoint['teacher']
        else:
            checkpoint_model = checkpoint['model']

        checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}
        
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)

        trunc_normal_(model.head.weight, std=2e-5)

    dataset_train = build_dataset(is_train='train', args=args)
    dataset_val = build_dataset(is_train='val', args=args)
    dataset_test = build_dataset(is_train='test', args=args)


    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if not args.eval:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
            if args.dist_eval:
                if len(dataset_val) % num_tasks != 0:
                    print(
                        'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank,
                    shuffle=True)  # shuffle=True to reduce monitor bias
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.task))
    else:
        log_writer = None

    if not args.eval:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

    # 2. 在main_finetune.py中，创建训练集加载器时添加加权采样器
    if not args.eval:  
    # 替换原数据集加载代码
#     if args.eval:
#         # 验证/测试集：无增强，无过采样
#         dataset_val = RetinaAugDataset(
#             data_path=args.data_path, split='val', 
#             input_size=args.input_size, is_train=False,
#             oversample_minority=False
#         )
#     else:
#         # 训练集：启用增强+过采样
#         dataset_train = RetinaAugDataset(
#             data_path=args.data_path, split='train', 
#             input_size=args.input_size, is_train=True,
#             oversample_minority=True  # 启用少数类过采样
#         )
        
#         # 1. 调用修正后的函数，获取权重（针对你的文件夹名称）
#         class_weights, target_class_order, class_counts = get_class_weights(args.data_path, split='train')
#         # 2. 获取数据集实际的“类别→标签”映射（关键！避免顺序错误）
#         # 示例：dataset_train.class_to_idx 可能为 {'depression':0, 'non-depression':1} 或相反
#         actual_class_to_idx = dataset_train.class_to_idx
#         print(f"数据集实际类别→标签映射: {actual_class_to_idx}")  # 必须打印确认！

#         # 3. 构建“标签→权重”的映射（核心：按实际标签匹配权重）
#         # target_class_order 是 [non-depression, depression]，对应权重 class_weights[0]、class_weights[1]
#         label_to_weight = {}
#         for cls, label in actual_class_to_idx.items():
#             if cls == target_class_order[0]:  # non-depression（多数类）
#                 label_to_weight[label] = class_weights[0]
#             elif cls == target_class_order[1]:  # depression（少数类）
#                 label_to_weight[label] = class_weights[1]
#         print(f"标签→权重映射: {label_to_weight}")  # 示例：{0:1.0, 1:5.0}（抑郁标签1对应高权重）

#         # 4. 为每个训练样本分配正确的权重
#         train_samples = dataset_train.samples  # 格式：[(img_path, label), ...]
#         train_labels = [label for _, label in train_samples]
#         sample_weights = torch.tensor([label_to_weight[label] for label in train_labels], dtype=torch.float)
#         print(f"前10个样本的权重: {sample_weights[:10]}")  # 确认少数类样本权重高

#         # 5. 初始化加权采样器（调整参数，避免过度重复）
#         sampler_train = torch.utils.data.WeightedRandomSampler(
#             weights=sample_weights,
#             num_samples=len(train_labels),  # 保持训练集总样本数不变
#             replacement=False  #避免少数类样本过度重复（关键！）    
#        )

#         # 6. 创建数据加载器（使用修正后的采样器）
#         data_loader_train = torch.utils.data.DataLoader(
#             dataset_train, sampler=sampler_train,
#             batch_size=args.batch_size,
#             num_workers=args.num_workers,  # 建议设为0或1，避免Windows共享内存问题
#             pin_memory=args.pin_mem,
#             drop_last=True,
#         )
        
        # 2. 替换损失函数初始化（替代原加权交叉熵）
#         loss_weights = torch.tensor([1.5039, 1.2767], dtype=torch.float).to(args.device)  # 抑郁:非抑郁权重
#         criterion = FocalLoss(alpha=1.0, gamma=4.0, class_weights=loss_weights)
#         print("启用Focal Loss，gamma=3.0，强化难分抑郁样本关注")
        
        print(f'len of train_set: {len(data_loader_train) * args.batch_size}')

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if args.resume and args.eval:
        checkpoint = torch.load(args.resume, map_location='cpu')
        print("Load checkpoint from: %s" % args.resume)
        model.load_state_dict(checkpoint['model'])

    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of model params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    no_weight_decay = model_without_ddp.no_weight_decay() if hasattr(model_without_ddp, 'no_weight_decay') else []
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=no_weight_decay,
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        if 'epoch' in checkpoint:
            print("Test with the best model at epoch = %d" % checkpoint['epoch'])
        test_stats, auc_roc = evaluate(data_loader_test, model, device, args, epoch=0, mode='test',
                                       num_class=args.nb_classes, log_writer=log_writer)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_score = 0.0
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        val_stats, val_score = evaluate(data_loader_val, model, device, args, epoch, mode='val',
                                        num_class=args.nb_classes, log_writer=log_writer)
        if max_score < val_score:
            max_score = val_score
            best_epoch = epoch
            if args.output_dir and args.savemodel:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, mode='best')
        print("Best epoch = %d, Best score = %.4f" % (best_epoch, max_score))


        if epoch == (args.epochs - 1):
            checkpoint = torch.load(os.path.join(args.output_dir, args.task, 'checkpoint-best.pth'), map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            model.to(device)
            print("Test with the best model, epoch = %d:" % checkpoint['epoch'])
            test_stats, auc_roc = evaluate(data_loader_test, model, device, args, -1, mode='test',
                                           num_class=args.nb_classes, log_writer=None)

        if log_writer is not None:
            log_writer.add_scalar('loss/val', val_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, args.task, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    criterion = torch.nn.CrossEntropyLoss()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, criterion)



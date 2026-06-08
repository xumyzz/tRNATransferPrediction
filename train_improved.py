#!/usr/bin/env python3
"""
改进版训练脚本 — 针对 cluster split 泛化问题
================================================
核心修改 (相对原 train_with_args.py):
  1. 损失函数: Focal+Dice+硬惩罚 → Distance-Aware Weighted BCE + Light Dice + Label Smoothing
  2. 训练策略: Family-Balanced Sampling + 课程学习
  3. 数据结构: 支持加载伪嵌合体 pickle 数据
  4. 域隔离: 预训练阶段默认关闭 (避免不必要的归纳偏置)

三阶段训练流程:
  Phase 1: bpRNA-1m 标准预训练 (domain isolation OFF, cluster split)
  Phase 2: bpRNA-1m + 伪嵌合体联合训练 (domain isolation ON, cluster split)
  Phase 3: 纯伪嵌合体精调 (可选)
"""

import os
import sys
import json
import random
import argparse
import pickle
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# ── 导入你的现有模块 ──
from src.config import Config
from src.dataset import MultiFileDatasetUpgrade, collate_pad
from src.model import SpotRNA_LSTM_Refined_BPPM_Chimeric
from scripts.cluster_utils import parse_cd_hit_clusters


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                  名称规范化 & 诊断                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def normalize_rna_name(raw: str) -> str:
    """
    统一 RNA 名称格式，消除 cd-hit .clstr 与 dataset.names 之间的差异。

    常见不匹配来源:
      - cd-hit: ">bpRNA_CRW_1..."  → 去掉 "> " 和 "..."
      - cd-hit: "bpRNA_CRW_1  *"   → 去掉尾部空格和 "*"
      - dataset: "bpRNA_CRW_1"     → 原样
    """
    name = raw.strip()
    # 去 > 前缀
    if name.startswith('>'):
        name = name[1:].strip()
    # 去 ... 后缀
    if name.endswith('...'):
        name = name[:-3].strip()
    # 去尾部空格 + 星号
    name = name.rstrip(' *')
    return name


def diagnose_name_mismatch(dataset_names: list, cluster_names: set, max_show: int = 10):
    """打印名称不匹配的诊断信息"""
    ds_names_set = set(dataset_names)
    nomatch_in_cluster = cluster_names - ds_names_set
    nomatch_in_dataset = ds_names_set - cluster_names

    print(f"\n[diagnose] 数据集名称数: {len(ds_names_set)}")
    print(f"[diagnose] cd-hit 名称数: {len(cluster_names)}")
    print(f"[diagnose] 交集: {len(ds_names_set & cluster_names)}")
    print(f"[diagnose] 仅在 cd-hit 中的名称 ({len(nomatch_in_cluster)}):")
    for n in sorted(list(nomatch_in_cluster))[:max_show]:
        print(f"    [{repr(n)[:80]}]")
    if len(nomatch_in_cluster) > max_show:
        print(f"    ... 还有 {len(nomatch_in_cluster) - max_show} 个")
    print(f"[diagnose] 仅在 dataset 中的名称 ({len(nomatch_in_dataset)}):")
    for n in sorted(list(nomatch_in_dataset))[:max_show]:
        print(f"    [{repr(n)[:80]}]")
    if len(nomatch_in_dataset) > max_show:
        print(f"    ... 还有 {len(nomatch_in_dataset) - max_show} 个")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                  改进版损失函数                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ImprovedRNALoss(nn.Module):
    """
    替代 ChimericRNALoss 的新损失函数。

    设计理念:
      - 去除 Focal Loss (在 cluster split 下会惩罚所有"难样本"，导致极端保守)
      - 去除硬约束惩罚 (diag/cross penalty 由模型 forward 中的 -1e9 硬掩码处理)
      - 距离感知正样本权重 → 关注长程配对 (tRNA 5'↔3' acceptor stem)
      - Label Smoothing → 防止过拟合家族指纹
      - 置信度 Margin → 鼓励模型输出 >0.6 的正样本置信度
    """

    def __init__(
        self,
        base_pos_weight: float = 2.0,
        w_dice: float = 0.1,
        w_margin: float = 0.05,
        margin_threshold: float = 0.6,
        label_smoothing: float = 0.05,
        distance_aware: bool = True,
        max_distance: int = 200,
    ):
        """
        Args:
            base_pos_weight: 基础正样本权重 (正负比例 ≈ 1:50-100)
            w_dice: Dice Loss 权重 (轻量，仅辅助茎区连续性)
            w_margin: 置信度 Margin Loss 权重 (鼓励自信预测)
            margin_threshold: 正样本期望最低置信度
            label_smoothing: 标签平滑系数
            distance_aware: 是否启用距离感知加权
            max_distance: 最大配对距离 (用于归一化)
        """
        super().__init__()
        self.base_pos_weight = base_pos_weight
        self.w_dice = w_dice
        self.w_margin = w_margin
        self.margin_threshold = margin_threshold
        self.label_smoothing = label_smoothing
        self.distance_aware = distance_aware
        self.max_distance = max_distance

    def forward(self, logits, targets, valid_mask, seg_ids=None):
        """
        Args:
            logits:    (B, L, L) 模型原始输出
            targets:   (B, L, L) 二值接触图
            valid_mask:(B, L, L) 有效位置 (排除 padding, 跨域, 对角线)
            seg_ids:   (B, L)     域标签 (0=tRNA, 2=insert)，可选
        """
        B, L, _ = logits.shape
        device = logits.device

        # ── 1. Label Smoothing ──
        targets_smooth = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        targets_smooth = targets_smooth * valid_mask
        targets_binary = targets * valid_mask  # 保留二值用于 margin

        # ── 2. 距离感知正样本权重矩阵 ──
        if self.distance_aware and L > 10:
            # 构建 (L, L) 距离矩阵
            pos_i = torch.arange(L, device=device).unsqueeze(1)  # (L, 1)
            pos_j = torch.arange(L, device=device).unsqueeze(0)  # (1, L)
            distance = (pos_i - pos_j).abs().float()              # (L, L)

            # 对数衰减权重: 越远配对权重越高 (鼓励长程预测)
            # tRNA 5'↔3' acceptor stem 距离 ~140nt
            dist_log = torch.log1p(distance / 10.0)               # (L, L)
            pos_weight_matrix = self.base_pos_weight * (1.0 + dist_log)  # (L, L)
        else:
            pos_weight_matrix = self.base_pos_weight * torch.ones(
                L, L, device=device
            )

        # ── 3. Weighted BCE ──
        bce = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, reduction='none'
        )  # (B, L, L)
        # 正样本用高权重，负样本用权重 1
        weight_map = torch.where(
            targets_binary > 0.5,
            pos_weight_matrix.unsqueeze(0),  # (1, L, L)
            torch.ones_like(bce),
        )
        bce_weighted = (bce * weight_map * valid_mask).sum()
        denom = (weight_map * valid_mask).sum().clamp(min=1.0)
        bce_loss = bce_weighted / denom

        # ── 4. Light Dice Loss ──
        if self.w_dice > 0:
            pred_prob = torch.sigmoid(logits) * valid_mask
            targets_masked = targets_binary
            intersection = (pred_prob * targets_masked).sum(dim=(-2, -1))
            pred_sum = pred_prob.sum(dim=(-2, -1))
            target_sum = targets_masked.sum(dim=(-2, -1))
            dice = 1.0 - (2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
            dice_loss = dice.mean()
        else:
            dice_loss = 0.0

        # ── 5. 置信度 Margin Loss (仅正样本) ──
        if self.w_margin > 0:
            pred_prob = torch.sigmoid(logits)
            # Margin Loss: 正样本概率如果 < margin_threshold, 施加惩罚
            pos_mask = (targets_binary > 0.5).float()
            margin = F.relu(self.margin_threshold - pred_prob) * pos_mask * valid_mask
            margin_loss = margin.sum() / (pos_mask * valid_mask).sum().clamp(min=1.0)
        else:
            margin_loss = 0.0

        # ── 6. 总损失 ──
        total_loss = bce_loss + self.w_dice * dice_loss + self.w_margin * margin_loss

        return total_loss, {
            'bce': bce_loss.item(),
            'dice': dice_loss.item() if isinstance(dice_loss, torch.Tensor) else dice_loss,
            'margin': margin_loss.item() if isinstance(margin_loss, torch.Tensor) else margin_loss,
            'total': total_loss.item(),
        }


def get_seg_ids(B, L, trna_5end_len, trna_3end_len, device):
    """
    三段式域标签：5'tRNA(域0) - miRNA前体(域2) - 3'tRNA(域0)

    预训练模式 (两端长度均为 None): 全序列同域 (全0)
    嵌合体模式: 三段 0-2-0
    """
    if trna_5end_len is None or trna_3end_len is None:
        return torch.zeros(B, L, dtype=torch.long, device=device)

    if isinstance(trna_5end_len, int):
        trna_5end_len = torch.full((B,), trna_5end_len, dtype=torch.long, device=device)
    if isinstance(trna_3end_len, int):
        trna_3end_len = torch.full((B,), trna_3end_len, dtype=torch.long, device=device)

    seg_ids = torch.zeros(B, L, dtype=torch.long, device=device)

    for b in range(B):
        five_len = trna_5end_len[b].item()
        three_len = trna_3end_len[b].item()
        if five_len + three_len >= L:
            continue
        mirna_start = five_len
        mirna_end = L - three_len
        if mirna_start < mirna_end:
            seg_ids[b, mirna_start:mirna_end] = 2

    return seg_ids


def build_valid_mask(masks, seg_ids, L, device):
    """
    构建 2D valid_mask:
      - 排除 padding 位置
      - 排除对角线
      - 排除跨域配对 (tRNA↔insert)

    返回值用于 loss 计算，与模型 forward 中的 -1e9 硬掩码一致。
    """
    B = masks.shape[0]

    # padding → 1D mask
    if masks.dim() == 4:
        mask_1d = masks[:, 0, 0, :]
    elif masks.dim() == 3:
        mask_1d = masks.squeeze(-1) if masks.shape[-1] == 1 else masks[:, :, 0]
    elif masks.dim() == 2:
        mask_1d = masks
    else:
        mask_1d = masks.view(B, -1)

    if mask_1d.dim() == 1:
        mask_1d = mask_1d.unsqueeze(0).expand(B, -1)

    # 2D padding mask
    valid = mask_1d.unsqueeze(2) * mask_1d.unsqueeze(1)  # (B, L, L)

    # 对角线排除
    diag = torch.eye(L, device=device).unsqueeze(0)        # (1, L, L)
    valid = valid * (1.0 - diag)

    # 跨域排除
    if seg_ids is not None:
        seg_i = seg_ids.unsqueeze(2)  # (B, L, 1)
        seg_j = seg_ids.unsqueeze(1)  # (B, 1, L)
        same_domain = (seg_i == seg_j).float()
        valid = valid * same_domain

    return valid.float()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                  Family-Balanced Sampling                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def create_family_balanced_sampler(dataset, clstr_path, seed=42):
    """
    基于 cd-hit 聚类结果构建 Family-Balanced Sampler。
    自动规范化名称以解决 cd-hit .clstr 与 dataset.names 间的格式差异。
    """
    random.seed(seed)
    np.random.seed(seed)

    clusters_raw = parse_cd_hit_clusters(clstr_path)
    print(f"[sampler] 加载 {len(clusters_raw)} 个家族")

    # ── 规范化所有 cluster 名称 ──
    clusters = [[normalize_rna_name(n) for n in cluster] for cluster in clusters_raw]

    # ── 处理 Subset ──
    if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
        underlying = dataset.dataset
        subset_indices = dataset.indices
    else:
        underlying = dataset
        subset_indices = list(range(len(dataset)))

    # ── 规范化 dataset 名称 ──
    ds_names_norm = [normalize_rna_name(n) for n in underlying.names]

    # ── 诊断(仅在丢失过多时打印) ──
    all_cluster_names = set()
    for cl in clusters:
        all_cluster_names.update(cl)
    ds_set = set(ds_names_norm)
    overlap = ds_set & all_cluster_names
    if len(overlap) < len(ds_set) * 0.5:
        diagnose_name_mismatch(ds_names_norm, all_cluster_names)

    # ── 构建 name → idx 映射 ──
    name_to_idx = {name: idx for idx, name in enumerate(ds_names_norm)}

    # ── 构建 idx → cluster 映射 ──
    idx_to_cluster = {}
    for ci, cluster in enumerate(clusters):
        for name in cluster:
            if name in name_to_idx:
                idx = name_to_idx[name]
                if idx not in idx_to_cluster:
                    idx_to_cluster[idx] = ci

    unmatched = set(subset_indices) - set(idx_to_cluster.keys())
    if unmatched:
        print(f"[sampler] {len(unmatched)}/{len(subset_indices)} 个样本未匹配任何 cd-hit 聚类")

    # ── 统计 + 权重 ──
    cluster_counts = defaultdict(int)
    for orig_idx in subset_indices:
        ci = idx_to_cluster.get(orig_idx, -1)
        cluster_counts[ci] += 1

    weights = []
    for orig_idx in subset_indices:
        ci = idx_to_cluster.get(orig_idx, -1)
        count = cluster_counts.get(ci, len(subset_indices))
        weight = 1.0 / max(count, 1)
        if ci == -1:
            weight = 1.0 / max(1, min(cluster_counts.values()) or 1)
        weights.append(weight)

    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float64),
        num_samples=len(subset_indices),
        replacement=True,
    )

    print(f"[sampler] 有效家族数: {len(cluster_counts)}")
    print(f"[sampler] 家族样本数范围: {min(cluster_counts.values())} - {max(cluster_counts.values())}")

    return sampler


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                  Cluster Split (复用原代码)                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def create_cluster_split(dataset, clstr_path, train_frac, val_frac, seed=42):
    """
    基于 cd-hit 聚类做 train/val/test split。
    自动规范化名称以匹配 dataset.names 与 .clstr 中的格式差异。
    未匹配到任何聚类的序列将被丢弃（避免 singleton 泄露）。
    """
    random.seed(seed)
    np.random.seed(seed)

    clusters_raw = parse_cd_hit_clusters(clstr_path)
    print(f"Found {len(clusters_raw)} clusters")

    # ── 规范化 ──
    clusters = [[normalize_rna_name(n) for n in cl] for cl in clusters_raw]
    ds_names_norm = [normalize_rna_name(n) for n in dataset.names]

    # ── 诊断 ──
    all_cluster_names = set()
    for cl in clusters:
        all_cluster_names.update(cl)
    overlap = set(ds_names_norm) & all_cluster_names
    if len(overlap) < len(ds_names_norm) * 0.5:
        diagnose_name_mismatch(ds_names_norm, all_cluster_names)

    # ── 匹配 ──
    name_to_idx = {name: idx for idx, name in enumerate(ds_names_norm)}
    cluster_indices = []
    matched_indices = set()
    missing_count = 0
    for cluster in clusters:
        indices = []
        for name in cluster:
            if name in name_to_idx:
                idx = name_to_idx[name]
                indices.append(idx)
                matched_indices.add(idx)
            else:
                missing_count += 1
        if indices:
            cluster_indices.append(indices)

    # ── 未匹配序列：丢弃（避免 singleton 泄露）──
    unmatched_indices = set(range(len(dataset))) - matched_indices
    if unmatched_indices:
        print(f"  Note: {len(unmatched_indices)} sequences not in any cd-hit cluster "
              f"— DISCARDED (would cause leakage if assigned as singletons)")
        # 不加入 cluster_indices，这些序列不会参与训练/验证

    if missing_count > 0:
        print(f"Warning: {missing_count} CD-HIT entries not matched to dataset "
              f"(dataset match rate: {len(overlap)/max(len(ds_names_norm),1):.1%})")
    print(f"Mapped to {len(cluster_indices)} families ({len(matched_indices)} matched, "
          f"{len(unmatched_indices)} discarded)")

    random.shuffle(cluster_indices)
    n_clusters = len(cluster_indices)
    n_train = int(n_clusters * train_frac)
    n_val = int(n_clusters * val_frac)
    train_clusters = cluster_indices[:n_train]
    val_clusters = cluster_indices[n_train:n_train + n_val]
    test_clusters = cluster_indices[n_train + n_val:]
    train_indices = [idx for cluster in train_clusters for idx in cluster]
    val_indices = [idx for cluster in val_clusters for idx in cluster]
    test_indices = [idx for cluster in test_clusters for idx in cluster]
    split_info = {
        'total_sequences': len(dataset),
        'matched_sequences': len(matched_indices),
        'discarded_sequences': len(unmatched_indices),
        'total_families': n_clusters,
        'train_families': len(train_clusters),
        'val_families': len(val_clusters),
        'test_families': len(test_clusters),
        'train_sequences': len(train_indices),
        'val_sequences': len(val_indices),
        'test_sequences': len(test_indices),
        'train_frac_actual': len(train_indices) / len(matched_indices),
        'val_frac_actual': len(val_indices) / len(matched_indices),
        'test_frac_actual': len(test_indices) / len(matched_indices),
    }
    return (train_indices, val_indices, test_indices), split_info


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                  主训练函数                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train_model(args):
    print("=" * 70)
    print("  改进版 RNA 二级结构训练 — Cluster Split 泛化优化")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Phase: {args.phase}")
    print(f"Configuration: {vars(args)}")

    # ── 设置随机种子 ──
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ═══════════════════════════════════════════════════════════
    # 1. 加载数据
    # ═══════════════════════════════════════════════════════════
    print(f"\n[1/6] Loading dataset from {args.data_dir}...")
    full_ds = MultiFileDatasetUpgrade(args.data_dir, max_len=args.max_len)
    if len(full_ds) == 0:
        print("Error: No data loaded. Check path.")
        return
    print(f"  Loaded {len(full_ds)} sequences")

    # ═══════════════════════════════════════════════════════════
    # 2. 数据集划分
    # ═══════════════════════════════════════════════════════════
    print(f"\n[2/6] Creating {'cluster' if args.clstr_path else 'random'} split...")

    if args.clstr_path:
        (train_idx, val_idx, test_idx), split_info = create_cluster_split(
            full_ds, args.clstr_path, args.train_frac, args.val_frac, args.seed
        )
        print("  Split info:")
        for key, value in split_info.items():
            print(f"    {key}: {value}")
        train_ds = Subset(full_ds, train_idx)
        val_ds = Subset(full_ds, val_idx)
    else:
        train_len = int(args.train_frac * len(full_ds))
        val_len = int(args.val_frac * len(full_ds))
        test_len = len(full_ds) - train_len - val_len
        from torch.utils.data import random_split
        train_ds, val_ds, _ = random_split(
            full_ds,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(args.seed)
        )
        print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {test_len}")

    # ═══════════════════════════════════════════════════════════
    # 3. 伪嵌合体数据加载 (可选)
    # ═══════════════════════════════════════════════════════════
    chimera_loader = None
    chimera_loader_iter = None

    if args.chimera_path and os.path.exists(args.chimera_path):
        print(f"\n[3/6] Loading pseudo-chimera data from {args.chimera_path}...")
        chimera_data = load_chimera_data(args.chimera_path)
        print(f"  Loaded {len(chimera_data)} pseudo-chimera samples")

        # chimera_only 模式：只用伪嵌合体训练，bpRNA 仅用于验证
        if args.phase == 'chimera_only':
            print(f"  Mode: CHIMERA ONLY — bpRNA used for validation only")
            args.domain_isolation = True
        elif args.phase == 'joint':
            print(f"  Mode: JOINT — bpRNA + chimera mixed training "
                  f"(chimera interleave=1/{args.chimera_interleave})")
    else:
        print(f"\n[3/6] No chimera data. (use --chimera_path to add)")
        chimera_data = None

    # ═══════════════════════════════════════════════════════════
    # 4. DataLoader
    # ═══════════════════════════════════════════════════════════
    print(f"\n[4/6] Building DataLoaders...")

    # ── bpRNA train ──
    if args.phase != 'chimera_only':
        if args.family_balance and args.clstr_path:
            print("  bpRNA: Family-Balanced Sampling")
            sampler = create_family_balanced_sampler(
                train_ds, args.clstr_path, seed=args.seed
            )
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size,
                sampler=sampler,
                collate_fn=collate_pad, num_workers=0,
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                collate_fn=collate_pad, num_workers=0,
            )
    else:
        # chimera_only: bpRNA 不用来训练
        train_loader = []
        print("  bpRNA: NOT used for training (chimera_only mode)")

    # ── bpRNA val (always used) ──
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_pad, num_workers=0,
    )

    # ── chimera DataLoader ──
    if chimera_data is not None:
        print(f"  Chimera: {len(chimera_data)} samples, batch_size={args.batch_size}")
        chimera_ds = ChimeraBPPMWrapper(chimera_data, max_len=args.max_len)
        chimera_loader = DataLoader(
            chimera_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=ChimeraBPPMWrapper.chimera_collate, num_workers=0,
        )
        # 对 chimera 域隔离固定开启
        args.domain_isolation = True

    # ═══════════════════════════════════════════════════════════
    # 5. 域隔离模式
    # ═══════════════════════════════════════════════════════════
    if args.domain_isolation:
        print(f"\n[5/6] Mode: CHIMERIC RNA — Domain Isolation ON")
        print(f"    5' tRNA: {args.trna_5end_len}nt, 3' tRNA: {args.trna_3end_len}nt")
    else:
        print(f"\n[5/6] Mode: STANDARD RNA — Domain Isolation OFF (推荐预训练)")
        # 预训练阶段强制关闭域隔离
        args.trna_5end_len = None
        args.trna_3end_len = None

    # ═══════════════════════════════════════════════════════════
    # 6. 模型 & 优化器 & 损失函数
    # ═══════════════════════════════════════════════════════════
    print(f"\n[6/6] Initializing model...")

    config = Config()
    config.RESNET_LAYERS = args.resnet_layers
    config.HIDDEN_DIM = args.hidden_dim
    config.LSTM_HIDDEN = args.lstm_hidden
    config.DEVICE = args.device

    model = SpotRNA_LSTM_Refined_BPPM_Chimeric(config).to(args.device)

    # ── 加载预训练权重 ──
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"  Loading pretrained weights from {args.pretrained_path}")
        try:
            state_dict = torch.load(args.pretrained_path, map_location=args.device)
            model.load_state_dict(state_dict, strict=False)
            print("  Weights loaded successfully!")
        except Exception as e:
            print(f"  Warning: Failed to load weights: {e}")

    # ── 优化器 ──
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── 学习率调度器 (Cosine Annealing with Warmup) ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── 新损失函数 ──
    criterion = ImprovedRNALoss(
        base_pos_weight=args.pos_weight,
        w_dice=args.w_dice,
        w_margin=args.w_margin,
        margin_threshold=args.margin_threshold,
        label_smoothing=args.label_smoothing,
        distance_aware=args.distance_aware,
        max_distance=args.max_len,
    )

    # ── 打印训练配置 ──
    print(f"\n  Model: {model.__class__.__name__}")
    print(f"  Loss: {criterion.__class__.__name__}")
    print(f"  Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})")
    print(f"  Scheduler: CosineAnnealing (min_lr={args.lr * 0.01:.2e})")
    print(f"  Base pos_weight: {args.pos_weight}")
    print(f"  Dice weight: {args.w_dice}")
    print(f"  Margin weight: {args.w_margin} (threshold: {args.margin_threshold})")
    print(f"  Label Smoothing: {args.label_smoothing}")
    print(f"  Distance-Aware: {args.distance_aware}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.accum_steps}")
    print(f"  Effective batch size: {args.batch_size * args.accum_steps}")

    # ═══════════════════════════════════════════════════════════
    # 训练循环
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  Starting training — {args.epochs} epochs")
    print(f"{'='*70}")

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_f1 = 0.0
    best_val_precision = 0.0
    best_val_recall = 0.0
    nan_count = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_bce = 0.0
        total_dice = 0.0
        total_margin = 0.0
        valid_batch_count = 0
        total_bpRNA_batches = 0
        total_chimera_batches = 0
        chim_step_count = 0  # chimera 独立梯度累积计数

        if chimera_loader is not None:
            chimera_loader_iter = iter(chimera_loader)

        for batch_idx, bp_batch in enumerate(train_loader):
            # ========================================================
            # Step A: bpRNA forward + backward (domain isolation OFF)
            # ========================================================
            seqs, bppms, labels, masks = bp_batch
            seqs = seqs.to(args.device)
            bppms = bppms.to(args.device)
            labels = labels.to(args.device)
            masks = masks.to(args.device)
            B, L, _ = seqs.shape

            logits = model(seqs, bppm=bppms, mask=masks,
                           trna_5end_len=None, trna_3end_len=None)
            seg_ids_loss = get_seg_ids(B, L, None, None, args.device)
            valid_mask = build_valid_mask(masks, seg_ids_loss, L, args.device)
            loss, loss_comps = criterion(logits, labels, valid_mask)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue

            loss = loss / args.accum_steps
            loss.backward()

            total_loss += loss_comps['total']
            total_bce += loss_comps['bce']
            total_dice += loss_comps['dice']
            total_margin += loss_comps['margin']
            valid_batch_count += 1
            total_bpRNA_batches += 1

            # bpRNA 梯度累积步进
            if total_bpRNA_batches % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # ========================================================
            # Step B: 插入 chimera batch（独立累积，独立 step）
            # ========================================================
            do_chim = (chimera_loader_iter is not None and
                       batch_idx % args.chimera_interleave == 0)

            if do_chim:
                try:
                    chim_batch = next(chimera_loader_iter)
                except StopIteration:
                    chimera_loader_iter = iter(chimera_loader)
                    chim_batch = next(chimera_loader_iter)

                c_seqs, c_bppms, c_labels, c_masks = chim_batch
                c_seqs = c_seqs.to(args.device)
                c_bppms = c_bppms.to(args.device)
                c_labels = c_labels.to(args.device)
                c_masks = c_masks.to(args.device)
                cB, cL, _ = c_seqs.shape

                c_logits = model(
                    c_seqs, bppm=c_bppms, mask=c_masks,
                    trna_5end_len=args.trna_5end_len,
                    trna_3end_len=args.trna_3end_len,
                )
                c_seg = get_seg_ids(cB, cL, args.trna_5end_len,
                                     args.trna_3end_len, args.device)
                c_valid = build_valid_mask(c_masks, c_seg, cL, args.device)
                c_loss, c_comps = criterion(c_logits, c_labels, c_valid)

                if torch.isnan(c_loss) or torch.isinf(c_loss):
                    nan_count += 1
                else:
                    # chimera 用自己的小 accum_steps（默认 2 个 chim batch 一 step）
                    chim_accum = max(1, args.accum_steps // 16)
                    c_loss = c_loss / chim_accum
                    c_loss.backward()
                    chim_step_count += 1

                    total_loss += c_comps['total']
                    total_bce += c_comps['bce']
                    total_dice += c_comps['dice']
                    total_margin += c_comps['margin']
                    valid_batch_count += 1
                    total_chimera_batches += 1

                    if chim_step_count % chim_accum == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        chim_step_count = 0

            # NaN 梯度检查
            has_nan_grad = False
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
            if has_nan_grad:
                nan_count += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            if batch_idx % 50 == 0:
                print(
                    f"  Epoch [{epoch+1}/{args.epochs}] "
                    f"Step [{batch_idx:4d}] "
                    f"Loss={loss_comps['total']:.4f} "
                    f"(BCE={loss_comps['bce']:.4f}, "
                    f"Dice={loss_comps['dice']:.4f}, "
                    f"Margin={loss_comps['margin']:.4f}) "
                    f"[bp={total_bpRNA_batches}, ch={total_chimera_batches}]"
                )

        # 清理 bpRNA 残余梯度
        if total_bpRNA_batches % args.accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        # 清理 chimera 残余梯度
        if chim_step_count > 0 and chimera_loader_iter is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # 学习率衰减
        scheduler.step()

        # ── Epoch 结束统计 ──
        avg_loss = total_loss / max(valid_batch_count, 1)
        avg_bce = total_bce / max(valid_batch_count, 1)
        avg_dice = total_dice / max(valid_batch_count, 1)
        avg_margin = total_margin / max(valid_batch_count, 1)

        print(
            f"\n{'─'*70}\n"
            f"  Epoch {epoch+1}/{args.epochs} Summary:\n"
            f"    Train Loss: {avg_loss:.4f} "
            f"(BCE={avg_bce:.4f} Dice={avg_dice:.4f} Margin={avg_margin:.4f})\n"
            f"    Batches: bpRNA={total_bpRNA_batches} chimera={total_chimera_batches}\n"
            f"    LR: {scheduler.get_last_lr()[0]:.2e}\n"
            f"    NaN skipped: {nan_count}"
        )

        # ═══════════════════════════════════════════════════════
        # Validation
        # ═══════════════════════════════════════════════════════
        model.eval()
        val_loss = 0.0
        val_bce = 0.0
        val_dice = 0.0
        val_f1_sum = 0.0
        val_p_sum = 0.0
        val_r_sum = 0.0
        val_batch_count = 0
        all_pred_probs = []

        with torch.no_grad():
            for batch_idx_val, (seqs, bppms, labels, masks) in enumerate(val_loader):
                seqs = seqs.to(args.device)
                bppms = bppms.to(args.device)
                labels = labels.to(args.device)
                masks = masks.to(args.device)
                B_val, L_val, _ = seqs.shape

                seg_ids_fwd = (
                    (args.trna_5end_len, args.trna_3end_len)
                    if args.domain_isolation else (None, None)
                )

                logits = model(
                    seqs,
                    bppm=bppms,
                    mask=masks,
                    trna_5end_len=seg_ids_fwd[0],
                    trna_3end_len=seg_ids_fwd[1],
                )

                seg_ids_val = get_seg_ids(
                    B_val, L_val,
                    args.trna_5end_len,
                    args.trna_3end_len,
                    args.device,
                )
                valid_mask_val = build_valid_mask(masks, seg_ids_val, L_val, args.device)

                loss, loss_comps = criterion(
                    logits, labels, valid_mask_val, seg_ids=seg_ids_val
                )
                val_loss += loss_comps['total']
                val_bce += loss_comps['bce']
                val_dice += loss_comps['dice']
                val_batch_count += 1

                # F1 计算
                probs = torch.sigmoid(logits)

                # 有效区域 mask (用于评估)
                if masks.dim() == 4:
                    mask_eval_1d = masks[:, 0, 0, :]
                elif masks.dim() == 2:
                    mask_eval_1d = masks
                else:
                    mask_eval_1d = masks.squeeze()
                mask_eval_2d = mask_eval_1d.unsqueeze(2) * mask_eval_1d.unsqueeze(1)
                mask_eval_2d = mask_eval_2d.view(B_val, L_val, L_val)

                # 仅取下三角
                tril = torch.tril(
                    torch.ones(L_val, L_val, device=args.device), diagonal=-1
                ).bool().unsqueeze(0)

                pred_binary = (probs > args.pp_offset).float() * mask_eval_2d * tril
                labels_eval = labels * mask_eval_2d * tril

                TP = (pred_binary * labels_eval).sum().item()
                FP = (pred_binary * (1 - labels_eval)).sum().item()
                FN = ((1 - pred_binary) * labels_eval).sum().item()

                p = TP / (TP + FP + 1e-8)
                r = TP / (TP + FN + 1e-8)
                f1 = 2 * p * r / (p + r + 1e-8)

                val_f1_sum += f1
                val_p_sum += p
                val_r_sum += r

                # 第一条 batch 的诊断信息
                if batch_idx_val == 0:
                    # 安全计算正样本概率（避免除零 NaN）
                    pos_mask = labels_eval > 0
                    pos_mean = probs[pos_mask].mean().item() if pos_mask.any() else float('nan')
                    nonzero_mask = mask_eval_2d > 0
                    print(
                        f"\n  {'='*20} Validation Diagnostics {'='*20}\n"
                        f"    Max prob: {probs.max().item():.4f}\n"
                        f"    Min prob (non-masked): "
                        f"{(probs[nonzero_mask]).min().item():.4f}\n"
                        f"    Mean prob (non-masked): "
                        f"{(probs[nonzero_mask]).mean().item():.4f}\n"
                        f"    Mean prob (pos sites): "
                        f"{pos_mean:.4f}\n"
                        f"    Raw F1: {f1:.4f} (P={p:.4f}, R={r:.4f})\n"
                    )

                    if args.domain_isolation:
                        cross_pred = (
                            (pred_binary > 0) &
                            (seg_ids_val.unsqueeze(2) != seg_ids_val.unsqueeze(1))
                        )
                        print(f"    Cross-domain false positives: {cross_pred.sum().item()} "
                              f"(should be 0)")
                    print(f"  {'='*52}")

        avg_val_loss = val_loss / max(val_batch_count, 1)
        avg_val_bce = val_bce / max(val_batch_count, 1)
        avg_val_dice = val_dice / max(val_batch_count, 1)
        avg_val_f1 = val_f1_sum / max(val_batch_count, 1)
        avg_val_p = val_p_sum / max(val_batch_count, 1)
        avg_val_r = val_r_sum / max(val_batch_count, 1)

        print(
            f"  Val Loss: {avg_val_loss:.4f} (BCE={avg_val_bce:.4f} "
            f"Dice={avg_val_dice:.4f})\n"
            f"  Val F1: {avg_val_f1:.4f}  |  "
            f"Precision: {avg_val_p:.4f}  |  Recall: {avg_val_r:.4f}\n"
            f"  pp_offset={args.pp_offset}"
        )

        # ── 保存模型 ──
        save_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1:03d}.pth")
        torch.save(model.state_dict(), save_path)

        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            best_val_precision = avg_val_p
            best_val_recall = avg_val_r
            best_path = os.path.join(args.save_dir, "model_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  ★ New best! F1={best_val_f1:.4f} "
                  f"(P={best_val_precision:.4f}, R={best_val_recall:.4f})")

    # ═══════════════════════════════════════════════════════════
    # 训练完成
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Best Val F1: {best_val_f1:.4f}")
    print(f"  Best Val P:  {best_val_precision:.4f}")
    print(f"  Best Val R:  {best_val_recall:.4f}")
    print(f"  Model saved to: {best_path}")
    print(f"{'='*70}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                  伪嵌合体数据加载器                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ChimeraBPPMWrapper:
    """
    将伪嵌合体 pickle 数据包装为兼容现有 DataLoader 的格式。

    伪嵌合体的 bppm 通道用全零填充 (或可选计算)，因为
    预训练阶段重点关注序列→结构的端到端学习。
    """

    def __init__(self, chimera_samples, max_len=200):
        self.chimeras = chimera_samples
        self.max_len = max_len
        self._index = list(range(len(self.chimeras)))

    def __len__(self):
        return len(self.chimeras)

    def __getitem__(self, idx):
        c = self.chimeras[idx]
        L = c.lengths['total']
        target_len = min(L, self.max_len)

        # 一键编码
        base_to_idx = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
        onehot = np.zeros((target_len, 4), dtype=np.float32)
        for i in range(target_len):
            if c.sequence[i] in base_to_idx:
                onehot[i, base_to_idx[c.sequence[i]]] = 1.0

        # contact map
        cmap = c.contact_map[:target_len, :target_len].copy()

        # bppm (dummy)
        bppm = np.zeros((target_len, target_len), dtype=np.float32)

        # mask (全1，因为无 padding)
        mask = np.ones((target_len, 1), dtype=np.float32)

        # domain labels
        domain = c.domain_labels[:target_len].copy()

        return (
            torch.tensor(onehot),
            torch.tensor(bppm),
            torch.tensor(cmap),
            torch.tensor(mask),
            torch.tensor(domain, dtype=torch.long),
        )

    @staticmethod
    def chimera_collate(batch):
        """
        自定义 collate: 处理不等长伪嵌合体的 padding。
        返回 4 元组 (seqs, bppms, labels, masks)，与原 collate_pad 兼容。
        """
        max_len = max(item[0].shape[0] for item in batch)
        B = len(batch)
        n_feat = batch[0][0].shape[1]

        padded_seqs = torch.zeros(B, max_len, n_feat)
        padded_bppms = torch.zeros(B, max_len, max_len)
        padded_labels = torch.zeros(B, max_len, max_len)
        padded_masks = torch.zeros(B, max_len, 1)

        for b, (seq, bppm, label, mask, domain) in enumerate(batch):
            L = seq.shape[0]
            padded_seqs[b, :L, :] = seq
            padded_bppms[b, :L, :L] = bppm
            padded_labels[b, :L, :L] = label
            padded_masks[b, :L, 0] = mask.squeeze()

        return padded_seqs, padded_bppms, padded_labels, padded_masks


def load_chimera_data(path):
    """加载伪嵌合体 pickle"""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # 自动识别格式
    from pseudo_chimera_generator import ChimeraSample

    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], ChimeraSample):
            return data
        elif isinstance(data[0], dict):
            # dict 格式 → ChimeraSample
            samples = []
            for d in data:
                samples.append(ChimeraSample(
                    sequence=d['sequence'],
                    contact_map=d['contact_map'],
                    domain_labels=d['domain_labels'],
                    trna_seq_5=d['trna_seq_5'],
                    trna_seq_3=d['trna_seq_3'],
                    insert_seq=d['insert_seq'],
                    trna_name=d['trna_name'],
                    insert_name=d['insert_name'],
                    lengths=d['lengths'],
                ))
            return samples

    raise ValueError(f"Unrecognized chimera data format: {type(data[0])}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                  命令行入口                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description='Improved RNA structure prediction training (Cluster Split Generalization)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 数据 ──
    parser.add_argument('--data_dir', type=str, required=True,
                        help='bpRNA 数据目录')
    parser.add_argument('--max_len', type=int, default=600)
    parser.add_argument('--clstr_path', type=str, default=None,
                        help='cd-hit 聚类结果路径 (用于 cluster split 和 family balance)')

    # ── 伪嵌合体数据 (新增) ──
    parser.add_argument('--chimera_path', type=str, default=None,
                        help='伪嵌合体 pickle 文件路径 (由 pseudo_chimera_generator.py 生成)')
    parser.add_argument('--chimera_ratio', type=float, default=0.3,
                        help='伪嵌合体混合比例 (已弃用，用 --chimera_interleave)')
    parser.add_argument('--chimera_interleave', type=int, default=20,
                        help='每隔 N 个 bpRNA batch 插入 1 个 chimera batch (默认: 20)')

    # ── 训练策略 ──
    parser.add_argument('--phase', type=str, default='pretrain',
                        choices=['pretrain', 'joint', 'chimera_only'],
                        help='训练阶段: pretrain(bpRNA only), joint(bpRNA+chimera), chimera_only')
    parser.add_argument('--domain_isolation', action='store_true',
                        help='启用域隔离 (预训练阶段应关闭, 联合训练/嵌合体训练时打开)')
    parser.add_argument('--family_balance', action='store_true',
                        help='启用 Family-Balanced Sampling (需要 --clstr_path)')
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--val_frac', type=float, default=0.1)

    # ── 模型参数 ──
    parser.add_argument('--resnet_layers', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lstm_hidden', type=int, default=64)
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='预训练权重路径 (Phase 2/3 从 Phase 1 的 best checkpoint 加载)')

    # ── 优化参数 ──
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--accum_steps', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # ── 损失函数参数 ──
    parser.add_argument('--pos_weight', type=float, default=2.0,
                        help='基础正样本权重 (建议 2.0-4.0)')
    parser.add_argument('--w_dice', type=float, default=0.1,
                        help='Dice Loss 权重 (轻量, 建议 0.05-0.2)')
    parser.add_argument('--w_margin', type=float, default=0.05,
                        help='Confidence Margin Loss 权重 (建议 0.01-0.1)')
    parser.add_argument('--margin_threshold', type=float, default=0.6,
                        help='正样本最低置信度阈值')
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                        help='标签平滑系数 (建议 0.025-0.10)')
    parser.add_argument('--distance_aware', type=int, default=1,
                        help='是否启用距离感知正样本权重 (1=是, 0=否)')

    # ── 域隔离参数 ──
    parser.add_argument('--trna_5end_len', type=int, default=43,
                        help="5' tRNA 前缀长度")
    parser.add_argument('--trna_3end_len', type=int, default=15,
                        help="3' tRNA 后缀长度")

    # ── 评估参数 ──
    parser.add_argument('--pp_offset', type=float, default=0.5,
                        help='二值化阈值')
    parser.add_argument('--pp_min_loop', type=int, default=4)

    # ── 其他 ──
    parser.add_argument('--save_dir', type=str, default='checkpoints_improved',
                        help='checkpoint 保存目录')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    # ── 设备 ──
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            args.device = torch.device(args.device)
        except Exception as e:
            print(f"Error: Invalid device '{args.device}': {e}")
            sys.exit(1)

    # ── 根据 phase 自动设置推荐参数 ──
    if args.phase == 'pretrain':
        # Phase 1: bpRNA 预训练
        print("\n" + "=" * 70)
        print("  Phase 1: bpRNA-1m Standard Pretraining")
        print("  Strategy: Cluster Split + Family-Balanced Sampling + Improved Loss")
        print("  Domain Isolation: OFF (推荐)")
        print("=" * 70)
        if not args.domain_isolation:
            print("  Note: Domain isolation is OFF — this is correct for pretraining.")
        if args.clstr_path and not args.family_balance:
            print("  Tip: Add --family_balance for rare family upsampling.")

    elif args.phase == 'joint':
        # Phase 2: bpRNA + 伪嵌合体联合训练
        print("\n" + "=" * 70)
        print("  Phase 2: bpRNA-1m + Pseudo-Chimera Joint Training")
        print("  Strategy: Load Phase 1 weights + Domain Isolation ON")
        print("=" * 70)
        if not args.pretrained_path:
            print("  Warning: --pretrained_path not set. Loading from scratch!")
        if not args.chimera_path:
            print("  Error: Phase 2 requires --chimera_path!")
            sys.exit(1)

    elif args.phase == 'chimera_only':
        # Phase 3: 纯伪嵌合体精调
        print("\n" + "=" * 70)
        print("  Phase 3: Pseudo-Chimera Only Fine-tuning")
        print("  Strategy: Load Phase 2 weights, train only on chimeras")
        print("=" * 70)
        if not args.pretrained_path:
            print("  Warning: --pretrained_path not set. Loading from scratch!")
        if not args.chimera_path:
            print("  Error: Phase 3 requires --chimera_path!")
            sys.exit(1)

    # ── 启动训练 ──
    train_model(args)


if __name__ == '__main__':
    main()

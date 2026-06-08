#!/usr/bin/env python3
"""
验证 singleton 伪家族是否造成数据泄露
用法:
    python diagnose_leakage.py \
        --data_dir /path/to/dbnFiles \
        --clstr_path /path/to/cd_hit.clstr \
        --model_path checkpoints_improved/model_best.pth \
        --device cuda
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.dataset import MultiFileDatasetUpgrade, collate_pad
from src.model import SpotRNA_LSTM_Refined_BPPM_Chimeric
from scripts.cluster_utils import parse_cd_hit_clusters


def normalize_rna_name(raw: str) -> str:
    name = raw.strip()
    if name.startswith('>'): name = name[1:].strip()
    if name.endswith('...'): name = name[:-3].strip()
    return name.rstrip(' *')


def create_split_with_tags(dataset, clstr_path, train_frac, val_frac, seed=42):
    """返回 train/val/test 索引 + 每条样本的标记(matched / singleton)"""
    random.seed(seed)
    np.random.seed(seed)

    clusters_raw = parse_cd_hit_clusters(clstr_path)
    clusters = [[normalize_rna_name(n) for n in cl] for cl in clusters_raw]
    ds_names_norm = [normalize_rna_name(n) for n in dataset.names]

    name_to_idx = {name: idx for idx, name in enumerate(ds_names_norm)}
    matched_indices = set()
    cluster_indices = []

    for cluster in clusters:
        indices = []
        for name in cluster:
            if name in name_to_idx:
                idx = name_to_idx[name]
                indices.append(idx)
                matched_indices.add(idx)
        if indices:
            cluster_indices.append(indices)

    # singleton 伪家族
    unmatched_indices = set(range(len(dataset))) - matched_indices
    singleton_clusters = []
    for idx in sorted(unmatched_indices):
        singleton_clusters.append([idx])

    # 分别 shuffle
    random.shuffle(cluster_indices)
    random.shuffle(singleton_clusters)

    # matched 按比例分
    n_matched = len(cluster_indices)
    n_m_train = int(n_matched * train_frac)
    n_m_val = int(n_matched * val_frac)

    train_matched = cluster_indices[:n_m_train]
    val_matched = cluster_indices[n_m_train:n_m_train + n_m_val]
    test_matched = cluster_indices[n_m_train + n_m_val:]

    # singleton 同样按比例分
    n_sing = len(singleton_clusters)
    n_s_train = int(n_sing * train_frac)
    n_s_val = int(n_sing * val_frac)

    train_sing = singleton_clusters[:n_s_train]
    val_sing = singleton_clusters[n_s_train:n_s_train + n_s_val]
    test_sing = singleton_clusters[n_s_train + n_s_val:]

    # 合并
    all_train = train_matched + train_sing
    all_val = val_matched + val_sing
    all_test = test_matched + test_sing

    train_indices = [idx for cl in all_train for idx in cl]
    val_indices = [idx for cl in all_val for idx in cl]
    test_indices = [idx for cl in all_test for idx in cl]

    # 标记每条 val 样本的来源
    val_matched_set = set()
    for cl in val_matched:
        val_matched_set.update(cl)
    val_sing_set = set()
    for cl in val_sing:
        val_sing_set.update(cl)

    val_tags = {}
    for idx in val_indices:
        if idx in val_matched_set:
            val_tags[idx] = 'matched'
        else:
            val_tags[idx] = 'singleton'

    return train_indices, val_indices, test_indices, val_tags


def compute_f1(logits, labels, masks, offset=0.5):
    """计算 batch 级别 F1（下三角 + mask 处理）"""
    B, L, _ = logits.shape
    probs = torch.sigmoid(logits)

    if masks.dim() == 4:
        m1d = masks[:, 0, 0, :]
    elif masks.dim() == 2:
        m1d = masks
    else:
        m1d = masks.squeeze()

    m2d = m1d.unsqueeze(2) * m1d.unsqueeze(1)
    m2d = m2d.view(B, L, L)

    tril = torch.tril(torch.ones(L, L, device=logits.device), diagonal=-1).bool().unsqueeze(0)

    preds = (probs > offset).float() * m2d * tril
    labels = labels * m2d * tril

    TP = (preds * labels).sum().item()
    FP = (preds * (1 - labels)).sum().item()
    FN = ((1 - preds) * labels).sum().item()

    p = TP / (TP + FP + 1e-8)
    r = TP / (TP + FN + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return f1, p, r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--clstr_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. 加载数据 & 划分
    print("[1/4] 加载数据...")
    full_ds = MultiFileDatasetUpgrade(args.data_dir, max_len=600)
    print(f"  总样本: {len(full_ds)}")

    train_idx, val_idx, test_idx, val_tags = create_split_with_tags(
        full_ds, args.clstr_path, train_frac=0.8, val_frac=0.1, seed=42
    )

    n_matched = sum(1 for t in val_tags.values() if t == 'matched')
    n_sing = sum(1 for t in val_tags.values() if t == 'singleton')
    print(f"  Val total: {len(val_idx)} (matched={n_matched}, singleton={n_sing})")

    # 2. 构建 val DataLoader（保持原始索引以区分来源）
    val_ds = Subset(full_ds, val_idx)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_pad, num_workers=0,
    )

    # 3. 加载模型
    print("[2/4] 加载模型...")
    config = Config()
    model = SpotRNA_LSTM_Refined_BPPM_Chimeric(config).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"  加载成功: {args.model_path}")

    # 4. 分别评估
    print("[3/4] 分别评估...")
    matched_f1_sum = 0.0
    matched_count = 0
    sing_f1_sum = 0.0
    sing_count = 0

    # val_idx → subset 内部索引映射
    idx_to_sub = {orig: sub for sub, orig in enumerate(val_idx)}

    with torch.no_grad():
        for batch_i, (seqs, bppms, labels, masks) in enumerate(val_loader):
            seqs = seqs.to(device)
            bppms = bppms.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            logits = model(seqs, bppm=bppms, mask=masks,
                           trna_5end_len=None, trna_3end_len=None)

            # 批内每条单独评估（避免一条的分母被另一条稀释）
            B = seqs.shape[0]
            for b in range(B):
                orig_idx = val_idx[batch_i * args.batch_size + b]
                tag = val_tags[orig_idx]

                single_logits = logits[b:b+1]
                single_labels = labels[b:b+1]
                single_masks = masks[b:b+1]

                f1, p, r = compute_f1(single_logits, single_labels, single_masks)

                if tag == 'matched':
                    matched_f1_sum += f1
                    matched_count += 1
                else:
                    sing_f1_sum += f1
                    sing_count += 1

    # 5. 报告
    print("\n[4/4] 结果")
    print("=" * 55)
    if matched_count > 0:
        matched_avg_f1 = matched_f1_sum / matched_count
        print(f"  cd-hit 匹配家族 Val F1:   {matched_avg_f1:.4f}  (n={matched_count})")
    if sing_count > 0:
        sing_avg_f1 = sing_f1_sum / sing_count
        print(f"  singleton 伪家族 Val F1:   {sing_avg_f1:.4f}  (n={sing_count})")
    print(f"  全体 Val F1:              {(matched_f1_sum + sing_f1_sum) / (matched_count + sing_count):.4f}")
    print("=" * 55)

    if 'matched_avg_f1' in dir() and 'sing_avg_f1' in dir():
        gap = sing_avg_f1 - matched_avg_f1
        if gap > 0.15:
            print(f"\n⚠️  singleton 比 cd-hit 高 {gap:.2f} → 存在明显泄露!")
            print("  建议: 去掉 singleton 伪家族, 只用 cd-hit 匹配的 ~4200 条训练")
        elif gap > 0.05:
            print(f"\n⚠️  轻微泄露 ({gap:.2f}), 影响可控但建议关注")
        else:
            print(f"\n✓ 差距 ({gap:.2f}) 在正常范围, 泄露不显著")

    if matched_count > 0:
        print(f"\n核心指标: cd-hit 匹配家族 F1 = {matched_avg_f1:.4f}")
        print(f"  这是最接近真实泛化能力的估计")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
嵌合体推理结果评测
===================
对比 inference_chimera.py 输出和 ground truth，分域报告 F1/P/R。

用法:
    python inference_chimera.py --pkl data/pseudo_chimera.pkl \
        --model checkpoints_v2_tr0_bppm_mfe/model_best.pth \
        --output_dir results/chimera_v2_bppm_mfe --device cuda:0

    python experiments/evaluate_chimera.py \
        --pred_dir results/chimera_v2_bppm_mfe \
        --gt_pkl data/pseudo_chimera.pkl
"""

import pickle, json, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def f1_score(pred, gt, mask):
    tp = ((pred > 0) & (gt > 0) & mask).sum()
    fp = ((pred > 0) & (gt == 0) & mask).sum()
    fn = ((pred == 0) & (gt > 0) & mask).sum()
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return f1, p, r, int(tp), int(fp), int(fn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred_dir', required=True)
    p.add_argument('--gt_pkl', required=True)
    args = p.parse_args()

    npy_files = sorted(Path(args.pred_dir).glob('*.npy'))
    with open(args.gt_pkl, 'rb') as f:
        gt_data = pickle.load(f)

    gt_idx = {}
    for i, item in enumerate(gt_data):
        name = item.get('trna_name', f'c_{i}') if isinstance(item, dict) else getattr(item, 'trna_name', f'c_{i}')
        gt_idx[name] = i

    stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    matched = skipped = 0

    for npy in npy_files:
        name = npy.stem
        if name not in gt_idx:
            skipped += 1
            continue
        matched += 1
        item = gt_data[gt_idx[name]]

        pred = np.load(npy)
        if isinstance(item, dict):
            gt = np.array(item['contact_map'])
            L = min(pred.shape[0], gt.shape[0], item['lengths']['total'])
            t5 = item['lengths'].get('trna_5', 43)
            t3 = item['lengths'].get('trna_3', 15)
        else:
            gt = item.contact_map
            L = min(pred.shape[0], gt.shape[0], item.lengths['total'])
            t5 = item.lengths.get('trna_5', 43)
            t3 = item.lengths.get('trna_3', 15)

        pred = pred[:L, :L]
        gt = gt[:L, :L]
        ins_s, ins_e = t5, L - t3

        # 总体（排除对角线）
        overall = np.ones((L, L), dtype=bool)
        np.fill_diagonal(overall, False)

        # 各域掩码
        masks = {
            'overall':  overall,
            "5'-tRNA":  _triu_mask(L, 0, ins_s, 0, ins_s),
            "3'-tRNA":  _triu_mask(L, ins_e, L, ins_e, L),
            'insert':   _triu_mask(L, ins_s, ins_e, ins_s, ins_e),
            "5'-3' cross": _cross_mask(L, ins_s, ins_e),
        }

        for k, m in masks.items():
            f1, p, r, tp, fp, fn = f1_score(pred, gt, m)
            s = stats[k]
            s['tp'] += tp
            s['fp'] += fp
            s['fn'] += fn

    # 打印
    print(f'\n预测文件: {len(npy_files)}  匹配: {matched}  跳过: {skipped}\n')
    print(f'{"":<16} {"F1":>8} {"P":>8} {"R":>8} {"TP":>6} {"FP":>6} {"FN":>6}')
    print('-' * 64)
    for k in ['overall', "5'-tRNA", "3'-tRNA", 'insert', "5'-3' cross"]:
        s = stats[k]
        tp, fp, fn = s['tp'], s['fp'], s['fn']
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        print(f'{k:<16} {f1:>8.4f} {p:>8.4f} {r:>8.4f} {tp:>6} {fp:>6} {fn:>6}')


def _triu_mask(L, r1, r2, c1, c2):
    m = np.zeros((L, L), dtype=bool)
    m[r1:r2, c1:c2] = True
    m &= np.triu(np.ones((L, L), dtype=bool), k=1)
    return m


def _cross_mask(L, ins_s, ins_e):
    m = np.zeros((L, L), dtype=bool)
    m[:ins_s, ins_e:] = True
    m[ins_e:, :ins_s] = True
    return m


if __name__ == '__main__':
    main()

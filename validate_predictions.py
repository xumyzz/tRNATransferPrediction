#!/usr/bin/env python3
"""
伪嵌合体预测效果验证
比对 predictions/*.npy vs chimeras.pkl 的 ground truth contact map

用法:
  python validate_predictions.py \
      --pred_dir ./chimera_predictions \
      --pkl ./data/chimeras.pkl \
      --trna_5end 43 --trna_3end 15
"""

import numpy as np
import argparse
import pickle
import os
import glob


def compute_f1(pred, gt, mask):
    """在 mask 有效区域内计算 F1"""
    pred = pred * mask
    gt = gt * mask

    TP = (pred * gt).sum()
    FP = (pred * (1 - gt)).sum()
    FN = ((1 - pred) * gt).sum()

    p = TP / (TP + FP + 1e-8)
    r = TP / (TP + FN + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return float(f1), float(p), float(r)


def analyze_sample(pred_cmap, gt_cmap, trna_5end=43, trna_3end=15):
    L = pred_cmap.shape[0]
    ins_s = trna_5end
    ins_e = L - trna_3end

    # 构建有效区域 mask（排除对角线 + 跨域）
    valid = np.ones((L, L), dtype=np.float32)
    np.fill_diagonal(valid, 0)

    # 跨域清零（tRNA ↔ insert 不应该配对）
    valid[:ins_s, ins_s:ins_e] = 0
    valid[ins_s:ins_e, :ins_s] = 0
    valid[ins_e:, ins_s:ins_e] = 0
    valid[ins_s:ins_e, ins_e:] = 0

    # 只取非对角线（上下三角都算）
    off_diag = np.ones((L, L), dtype=np.float32)
    np.fill_diagonal(off_diag, 0)
    valid = valid * off_diag

    # 全图 F1
    f1_all, p_all, r_all = compute_f1(pred_cmap, gt_cmap, valid)

    # 各区域 F1
    masks = {
        'tRNA5_internal': np.zeros((L, L)),
        'insert_internal': np.zeros((L, L)),
        'tRNA3_internal': np.zeros((L, L)),
        'cross_5_3':       np.zeros((L, L)),
    }
    masks['tRNA5_internal'][:ins_s, :ins_s] = 1
    masks['insert_internal'][ins_s:ins_e, ins_s:ins_e] = 1
    masks['tRNA3_internal'][ins_e:, ins_e:] = 1
    masks['cross_5_3'][:ins_s, ins_e:] = 1

    result = {'overall_f1': f1_all, 'overall_p': p_all, 'overall_r': r_all,
              'length': L}
    # 全图配对统计
    overall_valid = valid.copy()
    for k, v in masks.items():
        overall_valid += v
    overall_valid = (overall_valid > 0).astype(np.float32)
    pred_all = int((pred_cmap * overall_valid).sum())
    gt_all = int((gt_cmap * overall_valid).sum())
    result['overall_pred_pairs'] = pred_all
    result['overall_gt_pairs'] = gt_all

    for key, mask in masks.items():
        m = mask * valid  # 叠加通用 mask
        if m.sum() == 0:
            f1, p, r = 1.0, 1.0, 1.0 if gt_cmap[mask > 0].sum() == 0 else 0.0
        else:
            f1, p, r = compute_f1(pred_cmap, gt_cmap, m)
        result[key + '_f1'] = f1
        result[key + '_p'] = p
        result[key + '_r'] = r

        # 配对数量比较（在同一循环内）
        pred_n = int((pred_cmap * m).sum())
        gt_n = int((gt_cmap * m).sum())
        result[key + '_pred_pairs'] = pred_n
        result[key + '_gt_pairs'] = gt_n

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True, help='预测 .npy 目录')
    parser.add_argument('--pkl', type=str, required=True, help='伪嵌合体 ground truth')
    parser.add_argument('--trna_5end', type=int, default=43)
    parser.add_argument('--trna_3end', type=int, default=15)
    args = parser.parse_args()

    # 加载 GT
    with open(args.pkl, 'rb') as f:
        gt_data = pickle.load(f)

    # 构建 name → gt 映射
    gt_map = {}
    for item in gt_data:
        name = item.get('trna_name', '') if isinstance(item, dict) else getattr(item, 'trna_name', '')
        cmap = item['contact_map'] if isinstance(item, dict) else item.contact_map
        seq = item['sequence'] if isinstance(item, dict) else item.sequence
        gt_map[name] = {'cmap': cmap, 'seq': seq}

    # 扫描预测文件
    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, '*.npy')))
    print(f"预测文件: {len(pred_files)}")
    print(f"GT 条目:  {len(gt_map)}")

    all_stats = []
    matched = 0
    unmatched = 0

    for fpath in pred_files:
        fname = os.path.splitext(os.path.basename(fpath))[0]
        if fname not in gt_map:
            unmatched += 1
            continue
        matched += 1

        pred = np.load(fpath)
        gt = gt_map[fname]['cmap']

        # 对齐长度
        L = min(pred.shape[0], gt.shape[0])
        pred = pred[:L, :L]
        gt = gt[:L, :L]

        s = analyze_sample(pred, gt, args.trna_5end, args.trna_3end)
        s['name'] = fname
        all_stats.append(s)

    # ── 汇总 ──
    if not all_stats:
        print(f"\n[错误] 无匹配: matched={matched} unmatched={unmatched}")
        print(f"GT 名称示例: {list(gt_map.keys())[:5]}")
        print(f"预测文件示例: {[os.path.splitext(os.path.basename(f))[0] for f in pred_files[:5]]}")
        return

    keys = ['overall_f1', 'tRNA5_internal_f1', 'insert_internal_f1',
            'tRNA3_internal_f1', 'cross_5_3_f1']
    key_labels = ['Overall', "5' tRNA int", 'Insert int', "3' tRNA int", "5'↔3' cross"]

    print(f"\n{'='*60}")
    print(f"  匹配 {matched} 条 / 共 {len(pred_files)} 条预测")
    print(f"{'='*60}")
    print(f"  {'区域':<18} {'F1':>6} {'P':>6} {'R':>6}  {'Pred Pairs':>11}  {'GT Pairs':>9}")
    print(f"  {'-'*18} {'-'*6} {'-'*6} {'-'*6}  {'-'*11}  {'-'*9}")

    for key, label in zip(keys, key_labels):
        vals_f1 = [s[key] for s in all_stats]
        vals_p  = [s[key.replace('_f1', '_p')] for s in all_stats]
        vals_r  = [s[key.replace('_f1', '_r')] for s in all_stats]
        vals_pp = [s[key.replace('_f1', '_pred_pairs')] for s in all_stats]
        vals_gp = [s[key.replace('_f1', '_gt_pairs')] for s in all_stats]

        mean_f1 = np.mean(vals_f1)
        mean_p = np.mean(vals_p)
        mean_r = np.mean(vals_r)
        mean_pp = np.mean(vals_pp)
        mean_gp = np.mean(vals_gp)

        print(f"  {label:<18} {mean_f1:6.4f} {mean_p:6.4f} {mean_r:6.4f}  "
              f"{mean_pp:8.1f}/{len(vals_pp):>4}  {mean_gp:8.1f}/{len(vals_gp):>4}")

    print(f"  {'='*60}")

    # ── 低 F1 样本 ──
    all_stats.sort(key=lambda x: x['overall_f1'])
    print(f"\n  最低 5 条:")
    for s in all_stats[:5]:
        print(f"    {s['name'][:50]:<50} F1={s['overall_f1']:.4f} "
              f"P={s['overall_p']:.4f} R={s['overall_r']:.4f} "
              f"cross={s['cross_5_3_f1']:.4f}")

    # ── 保存 CSV ──
    csv_path = os.path.join(args.pred_dir, '_validation.csv')
    csv_keys = ['name', 'length'] + [k + '_f1' for k in keys] + [k + '_p' for k in keys] + [k + '_r' for k in keys]
    with open(csv_path, 'w') as f:
        f.write(','.join(csv_keys) + '\n')
        for s in all_stats:
            vals = []
            for k in csv_keys:
                v = s.get(k, 0)
                if isinstance(v, str):
                    vals.append(v)
                else:
                    vals.append(f"{v:.4f}")
            f.write(','.join(vals) + '\n')
    print(f"\n  CSV: {csv_path}")


if __name__ == '__main__':
    main()

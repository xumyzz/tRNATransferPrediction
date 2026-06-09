#!/usr/bin/env python3
"""
RNAfold 纯热力学基线 vs 模型预测

用法:
  python baseline_rnafold.py \
      --pkl ./data/chimeras.pkl \
      --save_predictions ./rnafold_predictions
"""

import numpy as np
import argparse
import pickle
import os
import glob


def dotbracket_to_contact(struct):
    """dot-bracket → (L, L) 对称 contact map"""
    L = len(struct)
    cmap = np.zeros((L, L), dtype=np.float32)
    stacks = {'(': []}
    for i, c in enumerate(struct):
        if c == '(':
            stacks['('].append(i)
        elif c == ')':
            if stacks['(']:
                j = stacks['('].pop()
                cmap[i, j] = 1.0
                cmap[j, i] = 1.0
    return cmap


def compute_f1(pred, gt, mask):
    """在 mask 有效区域内计算 F1"""
    TP = (pred * gt * mask).sum()
    FP = (pred * (1 - gt) * mask).sum()
    FN = ((1 - pred) * gt * mask).sum()
    p = TP / (TP + FP + 1e-8)
    r = TP / (TP + FN + 1e-8)
    return 2 * p * r / (p + r + 1e-8), p, r


def fold_single(seq):
    """ViennaRNA fold 一条序列，返回 (db_mfe, mfe_energy, db_cen)"""
    import RNA
    import traceback
    try:
        db_mfe, mfe = RNA.fold(seq)
        # centroid
        fc = RNA.fold_compound(seq)
        fc.pf()
        db_cen, _ = fc.centroid()
        return db_mfe, float(mfe), db_cen
    except Exception as e:
        print(f"  [WARN] RNAfold error on seq len={len(seq)}: {e}")
        traceback.print_exc()
        return '.' * len(seq), 0.0, '.' * len(seq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', type=str, required=True, help='chimeras.pkl')
    parser.add_argument('--model_pred_dir', type=str, default=None,
                        help='模型预测 .npy 目录（可选，自动找 ./chimera_predictions）')
    parser.add_argument('--save_predictions', type=str, default=None,
                        help='保存 RNAfold 预测 .npy 的目录')
    parser.add_argument('--fast', action='store_true',
                        help='只跑 MFE，跳过 centroid（快 10×）')
    parser.add_argument('--trna_5end', type=int, default=43)
    parser.add_argument('--trna_3end', type=int, default=15)
    args = parser.parse_args()

    ins_s = args.trna_5end

    # 加载 GT
    with open(args.pkl, 'rb') as f:
        gt_data = pickle.load(f)

    # 构建名 -> (cmap, seq) 映射
    gt_map = {}
    for item in gt_data:
        name = item.get('trna_name', '') if isinstance(item, dict) else getattr(item, 'trna_name', '')
        cmap = item['contact_map'] if isinstance(item, dict) else item.contact_map
        seq  = item['sequence'] if isinstance(item, dict) else item.sequence
        gt_map[name] = {'cmap': cmap, 'seq': seq}

    names = sorted(gt_map.keys())
    print(f"样本数: {len(names)}")

    # 收集模型预测（如果有）
    model_preds = {}
    if args.model_pred_dir:
        for fpath in glob.glob(os.path.join(args.model_pred_dir, '*.npy')):
            fname = os.path.splitext(os.path.basename(fpath))[0]
            if fname in gt_map:
                model_preds[fname] = np.load(fpath)
        print(f"模型预测: {len(model_preds)} 条")
    else:
        # 自动找
        for cand in ['./chimera_predictions', './chimera_predictions_v4',
                      './chimera_predictions_final']:
            if os.path.isdir(cand):
                for fpath in glob.glob(os.path.join(cand, '*.npy')):
                    fname = os.path.splitext(os.path.basename(fpath))[0]
                    if fname in gt_map and fname not in model_preds:
                        model_preds[fname] = np.load(fpath)
        if model_preds:
            print(f"模型预测: {len(model_preds)} 条 (自动检测)")
        else:
            print("(未找到模型预测，仅评估 RNAfold)")

    # 保存目录
    if args.save_predictions:
        os.makedirs(args.save_predictions, exist_ok=True)

    # ── RNAfold 预测 ──
    mode_str = "MFE only (--fast)" if args.fast else "MFE + centroid"
    print(f"\n运行 RNAfold ({mode_str})...")
    print(f"  样本数: {len(names)}")
    results_mfe  = []
    results_cen  = []
    results_mod  = []
    sample_details = []

    for ni, name in enumerate(names):
        gt_info = gt_map[name]
        seq  = gt_info['seq'][:600].upper().replace('T', 'U')
        gt_c = gt_info['cmap']
        L = min(len(seq), gt_c.shape[0])
        seq = seq[:L]
        gt_c = gt_c[:L, :L]
        ins_e = L - args.trna_3end

        # 构建 valid mask
        valid = np.ones((L, L), dtype=np.float32)
        np.fill_diagonal(valid, 0)
        valid[:ins_s, ins_s:ins_e] = 0
        valid[ins_s:ins_e, :ins_s] = 0
        valid[ins_e:, ins_s:ins_e] = 0
        valid[ins_s:ins_e, ins_e:] = 0

        # 域 mask
        mask_t5    = np.zeros((L, L)); mask_t5[:ins_s, :ins_s] = 1
        mask_ins   = np.zeros((L, L)); mask_ins[ins_s:ins_e, ins_s:ins_e] = 1
        mask_cross = np.zeros((L, L)); mask_cross[:ins_s, ins_e:] = 1
        for m in [mask_t5, mask_ins, mask_cross]:
            m *= valid

        # RNAfold
        try:
            if args.fast:
                import RNA
                db_mfe, mfe_e = RNA.fold(seq)
                db_cen = '.' * L
            else:
                db_mfe, mfe_e, db_cen = fold_single(seq)
        except Exception:
            db_mfe, db_cen = '.' * L, '.' * L

        cmap_mfe = dotbracket_to_contact(db_mfe)
        cmap_cen = dotbracket_to_contact(db_cen)

        # 保存
        if args.save_predictions:
            np.save(os.path.join(args.save_predictions, f"{name}_mfe.npy"), cmap_mfe)
            if not args.fast:
                np.save(os.path.join(args.save_predictions, f"{name}_cen.npy"), cmap_cen)

        # F1
        f1_all_mfe,_,_ = compute_f1(cmap_mfe, gt_c, valid)
        f1_all_cen,_,_ = compute_f1(cmap_cen, gt_c, valid)

        f1_t5_mfe,_,_   = compute_f1(cmap_mfe, gt_c, mask_t5)
        f1_ins_mfe,_,_  = compute_f1(cmap_mfe, gt_c, mask_ins)
        f1_cross_mfe,_,_ = compute_f1(cmap_mfe, gt_c, mask_cross)
        f1_t5_cen,_,_   = compute_f1(cmap_cen, gt_c, mask_t5)
        f1_ins_cen,_,_  = compute_f1(cmap_cen, gt_c, mask_ins)
        f1_cross_cen,_,_ = compute_f1(cmap_cen, gt_c, mask_cross)

        n_mfe  = int(cmap_mfe.sum() / 2)
        n_gt   = int((gt_c * valid).sum() / 2)

        results_mfe.append({
            'overall': f1_all_mfe, 't5': f1_t5_mfe,
            'ins': f1_ins_mfe, 'cross': f1_cross_mfe,
            'pairs': n_mfe,
        })
        results_cen.append({
            'overall': f1_all_cen, 't5': f1_t5_cen,
            'ins': f1_ins_cen, 'cross': f1_cross_cen,
            'pairs': int(cmap_cen.sum() / 2),
        })
        sample_details.append({
            'name': name, 'L': L, 'gt_pairs': n_gt,
            'mfe_pairs': n_mfe, 'cen_pairs': int(cmap_cen.sum() / 2),
            'mfe_f1': f1_all_mfe, 'cen_f1': f1_all_cen,
            'mfe_cross': f1_cross_mfe, 'mfe_ins': f1_ins_mfe,
        })

        # 模型预测对比
        if name in model_preds:
            mod_c = model_preds[name][:L, :L]
            f1_m,_,_ = compute_f1(mod_c, gt_c, valid)
            _,_,f1_m_ins = compute_f1(mod_c, gt_c, mask_ins)
            _,_,f1_m_cross = compute_f1(mod_c, gt_c, mask_cross)
            _,_,f1_m_t5 = compute_f1(mod_c, gt_c, mask_t5)
            results_mod.append({
                'overall': f1_m, 't5': f1_m_t5,
                'ins': f1_m_ins, 'cross': f1_m_cross,
                'pairs': int(mod_c.sum() / 2),
            })
        else:
            results_mod.append({'overall': 0, 't5': 0, 'ins': 0, 'cross': 0, 'pairs': 0})

        if ni == 0:
            print(f"  [sample 1] {name}: {L}nt, GT={int((gt_c*valid).sum()/2)}, "
                  f"MFE={int(cmap_mfe.sum()/2)}, cen={int(cmap_cen.sum()/2)}")

        if (ni + 1) % 100 == 0:
            print(f"  已处理 {ni+1}/{len(names)}")

    # ── 汇总 ──
    N = len(names)
    n_gt = np.mean([(gt_map[n]['cmap'].sum()/2) for n in names])
    n_mfe  = np.mean([r['pairs'] for r in results_mfe])
    n_cen  = np.mean([r['pairs'] for r in results_cen])
    n_mod  = np.mean([r['pairs'] for r in results_mod]) if results_mod else 0

    keys = ['overall', 't5', 'ins', 'cross']
    labels = ['Overall', "5' tRNA", 'Insert', "5'↔3'"]

    print(f"\n{'='*72}")
    print(f"  {N} samples  |  GT pairs={n_gt:.1f}")
    print(f"{'='*72}")
    print(f"  {'Method':<14} {'F1':>6} {'P':>6} {'R':>6}  {'Pairs':>8}  {'vs GT':>8}")
    print(f"  {'-'*14} {'-'*6} {'-'*6} {'-'*6}  {'-'*8}  {'-'*8}")

    for label, results in [('RNAfold MFE', results_mfe),
                            ('RNAfold cen.', results_cen)]:
        avg_f1 = np.mean([r['overall'] for r in results])
        avg_pairs = np.mean([r['pairs'] for r in results])
        avg_diff = avg_pairs - n_gt
        print(f"  {label:<14} {avg_f1:6.4f} {'--':>6} {'--':>6}  {avg_pairs:8.1f}  {avg_diff:+8.1f}")

    if results_mod:
        avg_f1 = np.mean([r['overall'] for r in results_mod])
        print(f"  {'Model (ours)':<14} {avg_f1:6.4f} {'--':>6} {'--':>6}  {n_mod:8.1f}  {n_mod-n_gt:+8.1f}")

    print()

    # ── 分域对比 ──
    print(f"  {'Domain':<14} {'RNAfold MFE':>12} {'RNAfold cen':>12} {'Model':>12}")
    print(f"  {'-'*14} {'-'*12} {'-'*12} {'-'*12}")
    for key, label in zip(keys, labels):
        mfe_val = np.mean([r[key] for r in results_mfe])
        cen_val = np.mean([r[key] for r in results_cen])
        mod_val = np.mean([r[key] for r in results_mod]) if results_mod else 0
        print(f"  {label:<14} {mfe_val:12.4f} {cen_val:12.4f} {mod_val:12.4f}")
    print(f"  {'='*72}")

    # ── 交叉配对专项 ──
    cross_gt  = np.mean([int((gt_map[n]['cmap'][:ins_s, gt_map[n]['cmap'].shape[1]-args.trna_3end:]).sum()) for n in names])
    cross_mfe = np.mean([r['cross'] for r in results_mfe])
    cross_cen = np.mean([r['cross'] for r in results_cen])

    print(f"\n  5'↔3' cross pairs — GT: {cross_gt:.1f}")
    print(f"    RNAfold MFE:   {cross_mfe:.4f}  (0 = 完全失败)")
    print(f"    RNAfold cen:   {cross_cen:.4f}")

    if results_mod:
        cross_mod = np.mean([r['cross'] for r in results_mod])
        print(f"    Model:         {cross_mod:.4f}")

    # ── Pair count 分布 ──
    print(f"\n  Pair count distribution:")
    for label, vals in [('GT', [(gt_map[n]['cmap'].sum()/2) for n in names]),
                          ('MFE', [r['pairs'] for r in results_mfe]),
                          ('cen', [r['pairs'] for r in results_cen])]:
        print(f"    {label:<6} mean={np.mean(vals):.1f} ±{np.std(vals):.1f}  "
              f"[{min(vals):.0f}, {max(vals):.0f}]")

    # ── 最好/最差 ──
    sample_details.sort(key=lambda x: x['mfe_f1'])
    print(f"\n  RNAfold MFE 最差 5 条:")
    for s in sample_details[:5]:
        print(f"    {s['name'][:50]:<50} F1={s['mfe_f1']:.4f}  cross={s['mfe_cross']:.4f}  pairs={s['mfe_pairs']}")

    print(f"\n  RNAfold MFE 最好 5 条:")
    for s in sample_details[-5:]:
        print(f"    {s['name'][:50]:<50} F1={s['mfe_f1']:.4f}  cross={s['mfe_cross']:.4f}  pairs={s['mfe_pairs']}")

    if args.save_predictions:
        print(f"\n  RNAfold 预测保存到: {args.save_predictions}/")


if __name__ == '__main__':
    main()

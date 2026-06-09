#!/usr/bin/env python3
"""嵌合体预测结果可视化 + 统计分析"""

import os, sys, glob
import numpy as np
import argparse

HAS_MPL = False
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    pass


def analyze_contact_map(cmap, trna_5end=43, trna_3end=15):
    L = cmap.shape[0]
    ins_start = trna_5end
    ins_end = L - trna_3end

    total = int(cmap.sum() / 2)
    t5_internal = int(cmap[:ins_start, :ins_start].sum() / 2)
    ins_internal = int(cmap[ins_start:ins_end, ins_start:ins_end].sum() / 2)
    t3_internal = int(cmap[ins_end:, ins_end:].sum() / 2)
    cross_5_3 = int(cmap[:ins_start, ins_end:].sum())
    cross_5_ins = int(cmap[:ins_start, ins_start:ins_end].sum())
    cross_ins_3 = int(cmap[ins_start:ins_end, ins_end:].sum())

    return {
        'total': total,
        'tRNA5_internal': t5_internal,
        'insert_internal': ins_internal,
        'tRNA3_internal': t3_internal,
        'cross_5_3': cross_5_3,
        'cross_5_insert': cross_5_ins,
        'cross_insert_3': cross_ins_3,
        'length': L,
    }


def visualize(cmap, name, out_path, trna_5end=43, trna_3end=15):
    if not HAS_MPL:
        return
    L = cmap.shape[0]
    ins_start = trna_5end
    ins_end = L - trna_3end

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.imshow(cmap, cmap='Blues', aspect='auto', interpolation='none', vmin=0, vmax=1)
    ax1.axhline(ins_start - 0.5, color='red', ls='--', lw=1)
    ax1.axhline(ins_end - 0.5, color='red', ls='--', lw=1)
    ax1.axvline(ins_start - 0.5, color='red', ls='--', lw=1)
    ax1.axvline(ins_end - 0.5, color='red', ls='--', lw=1)
    ax1.set_title(name[:60], fontsize=9)

    ax2.axis('off')
    s = analyze_contact_map(cmap, trna_5end, trna_3end)
    lines = [
        f"Length: {s['length']} nt",
        f"Total pairs: {s['total']}",
        f"5' tRNA: {s['tRNA5_internal']}    ins: {s['insert_internal']}    3': {s['tRNA3_internal']}",
        f"5'<->3' cross: {s['cross_5_3']}",
        f"Leak: 5'<->ins={s['cross_5_insert']}  ins<->3'={s['cross_insert_3']}",
    ]
    for i, line in enumerate(lines):
        ax2.text(0.05, 0.95 - i * 0.08, line, transform=ax2.transAxes,
                 fontsize=10, family='monospace')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--trna_5end', type=int, default=43)
    parser.add_argument('--trna_3end', type=int, default=15)
    parser.add_argument('--n_viz', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default='./viz_output')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not HAS_MPL:
        print("[注意] matplotlib 未安装, 输出文本报告 + CSV 替代图片")

    files = sorted(glob.glob(os.path.join(args.dir, '*.npy')))
    if not files:
        print(f"[错误] {args.dir} 下无 .npy 文件")
        return

    print(f"共 {len(files)} 个 .npy 文件")
    all_stats = []
    sample_lines = []

    for fi, fpath in enumerate(files):
        cmap = np.load(fpath)
        name = os.path.splitext(os.path.basename(fpath))[0]
        s = analyze_contact_map(cmap, args.trna_5end, args.trna_3end)
        s['name'] = name
        all_stats.append(s)

        if fi < args.n_viz:
            if HAS_MPL:
                visualize(cmap, name,
                          os.path.join(args.out_dir, f"{name}.png"),
                          args.trna_5end, args.trna_3end)
            else:
                pairs = list(zip(*np.where(np.triu(cmap) > 0)))
                t5_3 = [(i, j) for i, j in pairs
                        if i < args.trna_5end and j >= s['length'] - args.trna_3end]
                sample_lines.append(
                    f"\n{'='*50}\n"
                    f"[{fi+1}] {name} ({s['length']}nt)\n"
                    f"  total={s['total']}  tRNA5={s['tRNA5_internal']}  "
                    f"ins={s['insert_internal']}  tRNA3={s['tRNA3_internal']}\n"
                    f"  5'<->3' cross={s['cross_5_3']}\n"
                    f"  cross pairs: {t5_3}\n"
                    f"  leak: 5'<->ins={s['cross_5_insert']}  ins<->3'={s['cross_insert_3']}"
                )

    # 汇总
    totals   = [s['total'] for s in all_stats]
    crosses  = [s['cross_5_3'] for s in all_stats]
    leaks_5i = [s['cross_5_insert'] for s in all_stats]
    leaks_i3 = [s['cross_insert_3'] for s in all_stats]
    lengths  = [s['length'] for s in all_stats]

    summary = (
        f"\n{'='*55}\n"
        f"  {len(all_stats)} predictions\n"
        f"  Length:          {np.mean(lengths):.0f} [{min(lengths)}-{max(lengths)}]\n"
        f"  Total pairs:     {np.mean(totals):.1f} +-{np.std(totals):.1f}  [{min(totals)}-{max(totals)}]\n"
        f"  5'<->3' cross:   {np.mean(crosses):.1f} +-{np.std(crosses):.1f}  [{min(crosses)}-{max(crosses)}]\n"
        f"  cross ratio:     {sum(1 for c in crosses if c>0)/len(crosses):.1%}\n"
        f"  Leak 5'<->ins:   mean={np.mean(leaks_5i):.1f} max={max(leaks_5i)}\n"
        f"  Leak ins<->3':   mean={np.mean(leaks_i3):.1f} max={max(leaks_i3)}\n"
        f"{'='*55}"
    )
    print(summary)

    # 写报告
    rpt = os.path.join(args.out_dir, 'report.txt')
    with open(rpt, 'w') as f:
        f.write(summary)
        if sample_lines:
            f.write(f"\n\nTop {min(args.n_viz, len(files))} samples:\n")
            f.writelines(sample_lines)

    # 写 CSV
    csv = os.path.join(args.out_dir, 'stats.csv')
    keys = ['name','length','total','tRNA5_internal','insert_internal',
            'tRNA3_internal','cross_5_3','cross_5_insert','cross_insert_3']
    with open(csv, 'w') as f:
        f.write(','.join(keys) + '\n')
        for s in all_stats:
            f.write(','.join(str(s[k]) for k in keys) + '\n')

    print(f"report: {rpt}")
    print(f"csv:    {csv}")

    # 图表 (仅 matplotlib 可用时)
    if HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].hist(totals, bins=30, color='steelblue', edgecolor='white')
        axes[0].set_title('Total pairs')
        axes[1].hist(crosses, bins=30, color='darkorange', edgecolor='white')
        axes[1].set_title("5'<->3' cross")
        axes[2].hist(leaks_5i, bins=20, color='crimson', edgecolor='white')
        axes[2].set_title('Leak')
        plt.tight_layout()
        hist = os.path.join(args.out_dir, 'summary_hist.png')
        plt.savefig(hist, dpi=150)
        plt.close()
        print(f"hist:  {hist}")


if __name__ == '__main__':
    main()

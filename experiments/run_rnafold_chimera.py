#!/usr/bin/env python3
import pickle, argparse
from collections import defaultdict
import numpy as np
try:
    import RNA
except ImportError:
    print("pip install viennarna"); exit(1)

def dbn_to_contact(dbn, L):
    c = np.zeros((L, L), dtype=np.float32)
    stacks, pairs = {'(': []}, {')': '('}
    for i, ch in enumerate(dbn[:L]):
        if ch in stacks: stacks[ch].append(i)
        elif ch in pairs and stacks[pairs[ch]]:
            j = stacks[pairs[ch]].pop()
            c[i, j] = c[j, i] = 1.0
    return c

def f1(pred, gt, mask):
    tp = ((pred > 0) & (gt > 0) & mask).sum()
    fp = ((pred > 0) & (gt == 0) & mask).sum()
    fn = ((pred == 0) & (gt > 0) & mask).sum()
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    return 2 * p * r / (p + r + 1e-8), p, r, int(tp), int(fp), int(fn)

def _triu(L, r1, r2, c1, c2):
    m = np.zeros((L, L), dtype=bool)
    m[r1:r2, c1:c2] = True
    return m & np.triu(np.ones((L, L), dtype=bool), k=1)

def _cross(L, s, e):
    m = np.zeros((L, L), dtype=bool)
    m[:s, e:] = True; m[e:, :s] = True
    return m

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gt_pkl', required=True)
    p.add_argument('--max_samples', type=int, default=0)
    args = p.parse_args()

    with open(args.gt_pkl, 'rb') as f: data = pickle.load(f)
    if args.max_samples > 0: data = data[:args.max_samples]

    stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    ok = 0

    for i, item in enumerate(data):
        if isinstance(item, dict):
            seq = item['sequence']; gt = np.array(item['contact_map'])
            L = min(len(seq), gt.shape[0], item['lengths']['total'])
            t5 = item['lengths'].get('trna_5', 43)
            t3 = item['lengths'].get('trna_3', 15)
        else:
            seq = item.sequence; gt = item.contact_map
            L = min(len(seq), gt.shape[0], item.lengths['total'])
            t5 = item.lengths.get('trna_5', 43)
            t3 = item.lengths.get('trna_3', 15)

        seq = seq[:L].upper().replace('T', 'U').replace('N', 'A')
        gt = gt[:L, :L]

        try:
            rna_dbn, _ = RNA.fold(seq)  # ← 换成稳定 API
            pred = dbn_to_contact(rna_dbn, L)
        except Exception:
            continue
        ok += 1

        ins_s, ins_e = t5, L - t3
        overall = np.ones((L, L), dtype=bool)
        np.fill_diagonal(overall, False)

        masks = {
            'overall': overall,
            "5'-tRNA":  _triu(L, 0, ins_s, 0, ins_s),
            "3'-tRNA":  _triu(L, ins_e, L, ins_e, L),
            'insert':   _triu(L, ins_s, ins_e, ins_s, ins_e),
            "5'-3' cross": _cross(L, ins_s, ins_e),
        }
        for k, m in masks.items():
            _, _, _, tp, fp, fn = f1(pred, gt, m)
            s = stats[k]; s['tp'] += tp; s['fp'] += fp; s['fn'] += fn

        if (i + 1) % 200 == 0: print(f"  [{i+1}/{len(data)}]")

    print(f'\n样本数: {ok}\n')
    print(f'{"":<16} {"F1":>8} {"P":>8} {"R":>8} {"TP":>6} {"FP":>6} {"FN":>6}')
    print('-' * 64)
    for k in ['overall', "5'-tRNA", "3'-tRNA", 'insert', "5'-3' cross"]:
        s = stats[k]; tp, fp, fn = s['tp'], s['fp'], s['fn']
        p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
        print(f'{k:<16} {2*p*r/(p+r+1e-8):>8.4f} {p:>8.4f} {r:>8.4f} {tp:>6} {fp:>6} {fn:>6}')

if __name__ == '__main__':
    main()

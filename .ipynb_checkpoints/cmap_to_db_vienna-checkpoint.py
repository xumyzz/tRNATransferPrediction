#!/usr/bin/env python3
"""
Contact map → 纯净嵌套 () dot-bracket

跳过所有交叉配对（伪结），只保留严格嵌套的配对。
用法:
  python cmap_to_db_v2.py \
      --npy ./chimera_predictions/bpRNA_CRW_15563.npy \
      --pkl ./data/chimeras.pkl
"""

import numpy as np
import argparse
import pickle
import os


def contact_map_to_nested_db(cmap):
    """
    contact map → 纯嵌套 dot-bracket。
    按配对概率从高到低选，只保留不交叉的配对。
    """
    L = cmap.shape[0]
    # 取上三角所有配对，按概率降序
    pairs = []
    for i in range(L):
        for j in range(i + 1, L):
            if cmap[i, j] > 0:
                pairs.append((i, j, cmap[i, j]))
    pairs.sort(key=lambda x: -x[2])

    struct = ['.'] * L
    used = set()

    for a, b, _ in pairs:
        if a in used or b in used:
            continue
        # 检查是否与已分配的括号交叉
        conflict = False
        for u, v in [p for p in pairs if struct[p[0]] == '(']:
            if (a < u < b < v) or (u < a < v < b):
                conflict = True
                break
        if not conflict:
            struct[a] = '('
            struct[b] = ')'
            used.add(a)
            used.add(b)

    return ''.join(struct)


def find_sequence_in_pkl(pkl_path, npy_name):
    """从 pkl 匹配序列"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    key = os.path.splitext(os.path.basename(npy_name))[0]
    for item in data:
        name = item.get('trna_name', '') if isinstance(item, dict) else getattr(item, 'trna_name', '')
        if key in name:
            seq = item['sequence'] if isinstance(item, dict) else item.sequence
            return seq, name
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', type=str, required=True)
    parser.add_argument('--seq', type=str, default=None)
    parser.add_argument('--pkl', type=str, default=None)
    parser.add_argument('--idx', type=str, default=None)
    parser.add_argument('--trna_5end', type=int, default=43)
    parser.add_argument('--trna_3end', type=int, default=15)
    args = parser.parse_args()

    cmap = np.load(args.npy)
    L = cmap.shape[0]

    seq = args.seq
    name = None
    if seq is None and args.pkl:
        seq, name = find_sequence_in_pkl(args.pkl, args.idx or args.npy)
    if seq is None:
        print("[错误] 需要 --seq 或 --pkl")
        return
    seq = seq[:L].upper().replace('T', 'U')
    name = name or os.path.splitext(os.path.basename(args.npy))[0]

    db = contact_map_to_nested_db(cmap)
    n_pairs = db.count('(')
    # 同时统计如果允许交叉（原始 cmap_to_db），丢失了多少对
    orig_pairs = int(cmap.sum() / 2)
    lost = orig_pairs - n_pairs

    print(f">name: {name}  ({L}nt, {n_pairs} pairs", end="")
    if lost > 0:
        print(f", removed {lost} cross-pairs", end="")
    print(f")")

    npl = 60
    for k in range(0, L, npl):
        print(seq[k:k + npl])
    for k in range(0, L, npl):
        print(db[k:k + npl])

    ins_s = args.trna_5end
    ins_e = L - args.trna_3end
    dom = (ins_s * 'T' + (ins_e - ins_s) * 'I' + args.trna_3end * 'T')[:L]
    print("\n# domain: T=tRNA, I=insert")
    for k in range(0, L, npl):
        print(dom[k:k + npl])

    # 统计各域配对保留率
    t5_before = int(cmap[:ins_s, :ins_s].sum() / 2)
    cross_before = int(cmap[:ins_s, ins_e:].sum())
    ins_before = int(cmap[ins_s:ins_e, ins_s:ins_e].sum() / 2)
    t5_after = sum(1 for i in range(ins_s) for j in range(i+1, ins_s) if db[i]=='(' and db[j]==')')
    cross_after = sum(1 for i in range(ins_s) for j in range(ins_e, L) if db[i]=='(' and db[j]==')')
    ins_after = sum(1 for i in range(ins_s, ins_e) for j in range(i+1, ins_e) if db[i]=='(' and db[j]==')')
    print(f"\n# pair retention: 5'={t5_after}/{t5_before} cross={cross_after}/{cross_before} insert={ins_after}/{ins_before}")


if __name__ == '__main__':
    main()

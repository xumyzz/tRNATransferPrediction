#!/usr/bin/env python3
"""
Contact map → dot-bracket 结构转换 + 序列提取

用法:
  # 从 .pkl 找对应序列
  python cmap_to_db.py \
      --npy ./chimera_predictions/bpRNA_CRW_15563.npy \
      --pkl ./data/chimeras.pkl \
      --idx 0

  # 手动给序列
  python cmap_to_db.py \
      --npy ./chimera_predictions/bpRNA_CRW_15563.npy \
      --seq "GCGGAUUUA..."
"""

import numpy as np
import argparse
import pickle
import os


def contact_map_to_dotbracket(cmap):
    """
    contact map → dot-bracket 字符串。
    贪心回退: 按距离从近到远分配括号，优先用 ()。
    如果出现交叉则用 [] 表示伪结。
    """
    pairs = list(zip(*np.where(np.triu(cmap) > 0)))
    pairs = [(int(i), int(j)) for i, j in pairs]
    L = cmap.shape[0]
    struct = ['.'] * L

    if len(pairs) == 0:
        return ''.join(struct)

    # 按 i 升序，j 降序（嵌套优先）
    pairs.sort(key=lambda x: (x[0], -x[1]))

    # 贪心分配括号: 标为 '(' 的配对不能嵌套在不同括号里
    used = set()
    assigned = set()

    # 第一轮: 用 () 画所有不冲突的配对
    for a, b in pairs:
        if (a, b) in assigned:
            continue
        # 检查是否和已分配的 () 冲突
        conflict = False
        for u, v in [p for p in assigned if struct[p[0]] == '(']:
            # 嵌套关系: a<u<v<b 或 u<a<b<v → 同层
            if (a < u < v < b) or (u < a < b < v):
                continue
            # 交叉关系: a<u<b<v 或 u<a<v<b → 冲突
            if (a < u < b < v) or (u < a < v < b):
                conflict = True
                break
        if not conflict:
            struct[a] = '('
            struct[b] = ')'
            assigned.add((a, b))

    # 第二轮: 用 [] 画剩余的 (伪结)
    for a, b in pairs:
        if (a, b) in assigned:
            continue
        conflict = False
        for u, v in [p for p in assigned if struct[p[0]] == '[']:
            if (a < u < v < b) or (u < a < b < v):
                continue
            if (a < u < b < v) or (u < a < v < b):
                conflict = True
                break
        if not conflict:
            struct[a] = '['
            struct[b] = ']'
            assigned.add((a, b))

    # 第三轮: 用 {} 画剩余
    for a, b in pairs:
        if (a, b) in assigned:
            continue
        struct[a] = '{'
        struct[b] = '}'

    return ''.join(struct)


def find_sequence_in_pkl(pkl_path, npy_name_or_idx):
    """从 pkl 中找序列。npy_name_or_idx 可以是文件名关键词或索引号"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 按文件名匹配
    if isinstance(npy_name_or_idx, str):
        npy_name_or_idx = os.path.splitext(os.path.basename(npy_name_or_idx))[0]
        for i, item in enumerate(data):
            name = item.get('trna_name', '') if isinstance(item, dict) else getattr(item, 'trna_name', '')
            if npy_name_or_idx in name:
                seq = item['sequence'] if isinstance(item, dict) else item.sequence
                return seq, name
        return None, None

    # 按索引
    idx = int(npy_name_or_idx)
    item = data[idx]
    seq = item['sequence'] if isinstance(item, dict) else item.sequence
    name = item.get('trna_name', f'idx_{idx}') if isinstance(item, dict) else getattr(item, 'trna_name', f'idx_{idx}')
    return seq, name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', type=str, required=True, help='.npy contact map')
    parser.add_argument('--seq', type=str, default=None, help='序列 (不传则从 --pkl 找)')
    parser.add_argument('--pkl', type=str, default=None, help='chimeras.pkl')
    parser.add_argument('--idx', type=str, default=None, help='序列在 pkl 中的索引或名称关键词')
    parser.add_argument('--trna_5end', type=int, default=43)
    parser.add_argument('--trna_3end', type=int, default=15)
    args = parser.parse_args()

    cmap = np.load(args.npy)
    L = cmap.shape[0]

    # 获取序列
    seq = args.seq
    name = None
    if seq is None and args.pkl:
        seq, name = find_sequence_in_pkl(args.pkl, args.idx or args.npy)
    if seq is None:
        print("[错误] 需要 --seq 或 --pkl (含 --idx)")
        return

    seq = seq[:L].upper().replace('T', 'U')

    # 转换
    db = contact_map_to_dotbracket(cmap)

    n_pairs = int(cmap.sum() / 2)
    print(f">name: {name or 'unknown'}  ({L}nt, {n_pairs} pairs)")
    n_per_line = 60
    for k in range(0, L, n_per_line):
        print(seq[k:k + n_per_line])
    for k in range(0, L, n_per_line):
        print(db[k:k + n_per_line])

    # 输出包含域标记的注释行
    ins_s = args.trna_5end
    ins_e = L - args.trna_3end
    domain = (ins_s * 'T' + (ins_e - ins_s) * 'I' + args.trna_3end * 'T')[:L]
    print("\n# domain: T=tRNA, I=insert")
    for k in range(0, L, n_per_line):
        print(domain[k:k + n_per_line])


if __name__ == '__main__':
    main()

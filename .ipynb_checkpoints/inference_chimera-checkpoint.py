#!/usr/bin/env python3
"""
嵌合体 RNA 推理脚本
===================
用 Phase 1 预训练权重 + 后处理硬约束直接推理 contact map。

管线:
  模型输出 logits → 对角线/跨域清零 → AU/CG/GU 过滤
  → tRNA 模板软先验 → 最大权匹配 → 最终配对

用法:
  python inference_chimera.py \
      --seq "GCGGAUUUA..." \
      --model checkpoints_clean/model_best.pth \
      --output contact_map.npy

  # 批量推理
  python inference_chimera.py \
      --fasta chimeras.fasta \
      --model checkpoints_clean/model_best.pth \
      --output_dir ./predictions/
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from src.config import Config
from src.model import SpotRNA_LSTM_Refined_BPPM_Chimeric


# ═══════════════════════════════════════════════════════════════
# dot-bracket 解析
# ═══════════════════════════════════════════════════════════════

def dotbracket_to_pairs(struct: str):
    """dot-bracket → (i, j) 配对列表，仅处理 ()"""
    stacks = {'(': []}
    pairs = []
    for i, c in enumerate(struct):
        if c == '(':
            stacks['('].append(i)
        elif c == ')':
            if stacks['(']:
                j = stacks['('].pop()
                pairs.append((j, i))
    return pairs


# ═══════════════════════════════════════════════════════════════
# 序列 → one-hot
# ═══════════════════════════════════════════════════════════════

BASE_TO_IDX = {'A': 0, 'U': 1, 'G': 2, 'C': 3}

def seq_to_onehot(seq: str) -> np.ndarray:
    L = len(seq)
    x = np.zeros((L, 4), dtype=np.float32)
    for i, b in enumerate(seq.upper().replace('T', 'U')):
        if b in BASE_TO_IDX:
            x[i, BASE_TO_IDX[b]] = 1.0
    return x


# ═══════════════════════════════════════════════════════════════
# BPPM 计算 (ViennaRNA)
# ═══════════════════════════════════════════════════════════════

def compute_bppm(seq: str) -> np.ndarray:
    """用 ViennaRNA 计算配分函数 → BPPM 矩阵"""
    try:
        import RNA
        safe_seq = seq.upper().replace('T', 'U').replace('N', 'A')
        fc = RNA.fold_compound(safe_seq)
        fc.pf()
        bpp = fc.bpp()
        L = len(safe_seq)
        matrix = np.zeros((L, L), dtype=np.float32)
        for i in range(1, L + 1):
            for j in range(i + 1, L + 1):
                p = bpp[i][j]
                matrix[i-1, j-1] = p
                matrix[j-1, i-1] = p
        return matrix
    except ImportError:
        print("[警告] ViennaRNA 未安装，BPPM 用零矩阵代替")
        return np.zeros((len(seq), len(seq)), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# 后处理
# ═══════════════════════════════════════════════════════════════

CANONICAL_PAIRS = {('A','U'), ('U','A'), ('C','G'), ('G','C'), ('G','U'), ('U','G')}


def postprocess_contact_map(
    probs: np.ndarray,
    seq: str,
    trna_5end: int = 43,
    trna_3end: int = 15,
    use_template: bool = True,
    template_alpha: float = 0.3,
    allow_pseudoknot: bool = False,
    min_prob_t5: float = 0.15,
    min_prob_ins: float = 0.40,
    min_prob_cross: float = 0.10,
) -> np.ndarray:
    """
    后处理 contact map 概率矩阵。

    Args:
        ...
        min_prob_t5:    5' tRNA 域最低配对概率
        min_prob_ins:   insert 域最低配对概率 (推荐设高，减少误报)
        min_prob_cross: 跨域配对最低概率
    """
    L = len(seq)
    seq_upper = seq.upper().replace('T', 'U')

    # 对角线 + 对称化 + 跨域清零 + AU/CG/GU + 模板（同上）
    np.fill_diagonal(probs, 0.0)
    probs = (probs + probs.T) / 2.0

    ins_start = trna_5end
    ins_end = L - trna_3end
    probs[:ins_start, ins_start:ins_end] = 0.0
    probs[ins_start:ins_end, :ins_start] = 0.0
    probs[ins_end:, ins_start:ins_end] = 0.0
    probs[ins_start:ins_end, ins_end:] = 0.0

    for i in range(L):
        for j in range(i + 1, L):
            if (seq_upper[i], seq_upper[j]) not in CANONICAL_PAIRS:
                probs[i, j] = 0.0
                probs[j, i] = 0.0

    if use_template:
        trna_template = build_trna_template(trna_5end, ins_end - ins_start, trna_3end)
        for i in range(L):
            for j in range(i + 1, L):
                both_trna = ((i < ins_start or i >= ins_end) and
                             (j < ins_start or j >= ins_end))
                if both_trna and trna_template[i, j] > 0:
                    probs[i, j] = (1 - template_alpha) * probs[i, j] + template_alpha * 1.0
                    probs[j, i] = probs[i, j]

    # ── 分域最小概率过滤 ──
    for i in range(L):
        for j in range(i + 1, L):
            p = probs[i, j]
            # 判断所属域
            in_t5 = (i < ins_start and j < ins_start)
            in_ins = (ins_start <= i < ins_end and ins_start <= j < ins_end)
            is_cross = (i < ins_start and j >= ins_end)
            if in_ins and p < min_prob_ins:
                probs[i, j] = 0.0; probs[j, i] = 0.0
            elif in_t5 and p < min_prob_t5:
                probs[i, j] = 0.0; probs[j, i] = 0.0
            elif is_cross and p < min_prob_cross:
                probs[i, j] = 0.0; probs[j, i] = 0.0

    # ── Insert 域：ViennaRNA MFE 折叠 ──
    ins_seq = seq_upper[ins_start:ins_end]
    if len(ins_seq) > 2:
        try:
            import RNA
            ins_db, _ = RNA.fold(ins_seq)
            for ci, cj in dotbracket_to_pairs(ins_db):
                i = ins_start + ci
                j = ins_start + cj
                if 0 <= i < L and 0 <= j < L:
                    probs[i, j] = 1.0
                    probs[j, i] = 1.0
        except Exception:
            pass

    # 贪心匹配（tRNA 域 + 跨域，insert 域已经用 Vienna 锁定）
    pairs = greedy_max_weight_matching(probs, max_pairs_per_base=1)
    contact = np.zeros((L, L), dtype=np.float32)
    for i, j in pairs:
        contact[i, j] = 1.0
        contact[j, i] = 1.0
    return contact


def build_trna_template(L5: int, ins_len: int, L3: int) -> np.ndarray:
    """
    构建 tRNA 结构模板。基于标准 tRNA 三叶草结构的配对位置。
    L5 = 5' tRNA 长度, ins_len = insert 长度, L3 = 3' tRNA 长度
    """
    total = L5 + ins_len + L3
    template = np.zeros((total, total), dtype=np.float32)

    # 标准 tRNA 配对: 5' 端第 n 个碱基 ↔ 3' 端倒数第 n 个碱基
    # 为 acceptor stem (最多 7 bp)
    stem_len = min(7, L5, L3)
    for k in range(stem_len):
        i = k                    # 5' 端
        j = L5 + ins_len + (L3 - 1 - k)  # 3' 端
        if 0 <= i < total and 0 <= j < total:
            template[i, j] = 1.0
            template[j, i] = 1.0

    return template


def greedy_max_weight_matching(probs: np.ndarray, max_pairs_per_base: int = 1):
    """
    贪心最大权匹配。
    按概率从高到低逐个选配对，跳过已配对或冲突的位置。
    """
    L = probs.shape[0]
    # 只取上三角
    indices = np.triu_indices(L, k=1)
    scores = probs[indices]
    order = np.argsort(-scores)

    used = np.zeros(L, dtype=int)
    pairs = []

    for idx in order:
        i, j = int(indices[0][idx]), int(indices[1][idx])
        s = scores[idx]
        if s <= 0:
            continue
        if used[i] >= max_pairs_per_base or used[j] >= max_pairs_per_base:
            continue
        used[i] += 1
        used[j] += 1
        pairs.append((i, j))

    return pairs


# ═══════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════

def load_model(weight_path: str, device: str = 'cpu'):
    config = Config()
    model = SpotRNA_LSTM_Refined_BPPM_Chimeric(config).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════
# 推理
# ═══════════════════════════════════════════════════════════════

def predict_chimera(
    model,
    seq: str,
    trna_5end: int = 43,
    trna_3end: int = 15,
    device: str = 'cpu',
    return_probs: bool = False,
    template_alpha: float = 0.3,
    min_prob_t5: float = 0.15,
    min_prob_ins: float = 0.40,
    min_prob_cross: float = 0.10,
):
    """对一条嵌合体序列做完整推理"""
    L = len(seq)

    # one-hot
    onehot = seq_to_onehot(seq)
    # BPPM
    bppm = compute_bppm(seq)
    # mask: (L,) → 和训练时 collate_pad 的 masks 格式一致
    mask = np.ones(L, dtype=np.float32)

    # 转 tensor → (1, L, 4), (1, L, L), (1, L)
    x = torch.tensor(onehot).unsqueeze(0).to(device)
    b = torch.tensor(bppm).unsqueeze(0).to(device)
    m = torch.tensor(mask).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(
            x, bppm=b, mask=m,
            trna_5end_len=trna_5end,
            trna_3end_len=trna_3end,
        )

    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    # 后处理
    contact = postprocess_contact_map(
        probs, seq,
        trna_5end=trna_5end,
        trna_3end=trna_3end,
        use_template=True,
        template_alpha=template_alpha,
        min_prob_t5=min_prob_t5,
        min_prob_ins=min_prob_ins,
        min_prob_cross=min_prob_cross,
    )

    if return_probs:
        return contact, probs
    return contact


# ═══════════════════════════════════════════════════════════════
# 命令行
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='嵌合体 RNA 推理')
    parser.add_argument('--seq', type=str, default=None,
                        help='单条嵌合体序列')
    parser.add_argument('--fasta', type=str, default=None,
                        help='FASTA 格式的多条序列 (支持 .gz)')
    parser.add_argument('--pkl', type=str, default=None,
                        help='伪嵌合体 pickle 文件 (直接从 chimeras.pkl 推理)')
    parser.add_argument('--model', type=str, required=True,
                        help='Phase 1 最佳权重路径')
    parser.add_argument('--output', type=str, default='contact.npy',
                        help='输出路径 (单条 .npy, 批量目录)')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='批量输出目录')
    parser.add_argument('--trna_5end', type=int, default=43)
    parser.add_argument('--trna_3end', type=int, default=15)
    parser.add_argument('--template_alpha', type=float, default=0.3,
                        help='tRNA 模板权重 (0=纯模型, 1=纯模板)')
    parser.add_argument('--min_prob_t5', type=float, default=0.08,
                        help="5' tRNA 域最低配对概率 (默认: 0.08)")
    parser.add_argument('--min_prob_ins', type=float, default=0.50,
                        help='Insert 域最低配对概率 (仅对模型预测部分生效)')
    parser.add_argument('--min_prob_cross', type=float, default=0.06,
                        help='跨域配对最低概率 (默认: 0.06)')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.seq is None and args.fasta is None and args.pkl is None:
        print("[错误] 需要 --seq / --fasta / --pkl 之一")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[设备] {device}")

    model = load_model(args.model, device)
    print(f"[模型] 加载: {args.model}")

    # 单条推理
    if args.seq:
        seq = args.seq.upper().replace('T', 'U')
        print(f"[序列] 长度 {len(seq)}")

        contact = predict_chimera(
            model, seq,
            trna_5end=args.trna_5end,
            trna_3end=args.trna_3end,
            device=device,
            template_alpha=args.template_alpha,
            min_prob_t5=args.min_prob_t5,
            min_prob_ins=args.min_prob_ins,
            min_prob_cross=args.min_prob_cross,
        )

        np.save(args.output, contact)
        n_pairs = int(contact.sum() / 2)
        print(f"[结果] {n_pairs} 对配对 → {args.output}")

    # 批量 FASTA
    if args.fasta:
        os.makedirs(args.output_dir, exist_ok=True)
        sequences = parse_fasta(args.fasta)
        print(f"[批量] {len(sequences)} 条序列")

        for name, seq in sequences.items():
            seq = seq.upper().replace('T', 'U')
            contact = predict_chimera(
                model, seq,
                trna_5end=args.trna_5end,
                trna_3end=args.trna_3end,
                device=device,
                template_alpha=args.template_alpha,
                min_prob_t5=args.min_prob_t5,
                min_prob_ins=args.min_prob_ins,
                min_prob_cross=args.min_prob_cross,
            )
            out_path = os.path.join(args.output_dir, f"{name}.npy")
            np.save(out_path, contact)

        print(f"[完成] 结果保存到 {args.output_dir}/")

    # 批量 pkl
    if args.pkl:
        import pickle
        os.makedirs(args.output_dir, exist_ok=True)
        with open(args.pkl, 'rb') as f:
            data = pickle.load(f)
        print(f"[pkl] {len(data)} 条嵌合体")

        for i, item in enumerate(data):
            # 兼容 dict / ChimeraSample
            if hasattr(item, 'sequence'):
                seq = item.sequence.upper().replace('T', 'U')
                name = getattr(item, 'trna_name', f'chimera_{i}')
            else:
                seq = item['sequence'].upper().replace('T', 'U')
                name = item.get('trna_name', f'chimera_{i}')

            # 序列太长截断
            if len(seq) > 600:
                seq = seq[:600]

            contact = predict_chimera(
                model, seq,
                trna_5end=args.trna_5end,
                trna_3end=args.trna_3end,
                device=device,
                template_alpha=args.template_alpha,
                min_prob_t5=args.min_prob_t5,
                min_prob_ins=args.min_prob_ins,
                min_prob_cross=args.min_prob_cross,
            )
            safe_name = name.replace('/', '_').replace('\\', '_')
            out_path = os.path.join(args.output_dir, f"{safe_name}.npy")
            np.save(out_path, contact)

            if (i + 1) % 100 == 0:
                print(f"  已处理 {i+1}/{len(data)}")

        print(f"[完成] {len(data)} 条 → {args.output_dir}/")


def parse_fasta(path: str) -> dict:
    """解析 FASTA 文件 → {name: seq}，自动处理 gzip 压缩"""
    result = {}
    import gzip

    # 自动检测 gzip
    with open(path, 'rb') as fb:
        magic = fb.read(2)
    is_gzip = (magic == b'\x1f\x8b')

    opener = gzip.open if is_gzip else open
    with opener(path, 'rt', encoding='utf-8') as f:
        name = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name:
                    result[name] = ''.join(seq_lines)
                name = line[1:].split()[0]
                seq_lines = []
            else:
                seq_lines.append(line)
        if name:
            result[name] = ''.join(seq_lines)
    return result


if __name__ == '__main__':
    main()

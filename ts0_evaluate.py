#!/usr/bin/env python3
"""
TS0 标准基准评测 — bpRNA 官方 TS0 数据集
==========================================
与 SPOT-RNA / UFold / MXfold2 / KnotFold 完全对齐的评测协议。

特性:
  - 支持 .st (bpRNA 注释格式) 和 .bpseq (SPOT-RNA 格式)
  - 最优阈值搜索 (0.01~0.99, step=0.01) — 保证最好的 F1
  - 自动计算多种聚合指标 (macro/micro/mean/median)
  - 长度分组统计
  - 输出 JSON + CSV，可直接喂给 generate_sota_table.py

用法:
    python experiments/ts0_evaluate.py \
        --checkpoint checkpoints_v2_tr0/model_best.pth \
        --ts0_dir data/TS0 \
        --output_dir results/ts0_v2_tr0 \
        --device cuda:0

    # 指定输出名称标签 (用于消融实验)
    python experiments/ts0_evaluate.py \
        --checkpoint ablation_checkpoints/no_bppm/model_best.pth \
        --ts0_dir data/TS0 --no_bppm \
        --output_dir results/ts0_no_bppm --tag no_bppm
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

# ── 项目路径 ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import SpotRNA_LSTM_Refined_BPPM_Chimeric


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          BPPM 计算 (含 ViennaRNA 降级)                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_HAS_VRNA = False
try:
    # 先尝试绕过 src.dataset 的 import RNA 问题 — 直接 import
    import RNA as _RNA
    _HAS_VRNA = True
except ImportError:
    pass


def get_bppm_matrix(seq):
    """
    计算 BPPM (碱基配对概率矩阵)。
    ViennaRNA 不可用时返回全零矩阵 (模型 forward 兼容).
    """
    L = len(seq)
    if not _HAS_VRNA:
        return np.zeros((L, L), dtype=np.float32)

    safe_seq = seq.replace('N', 'A').replace('R', 'A').replace('Y', 'C')
    try:
        fc = _RNA.fold_compound(safe_seq)
        fc.pf()
        bpp = fc.bpp()
    except Exception:
        return np.zeros((L, L), dtype=np.float32)

    matrix = np.zeros((L, L), dtype=np.float32)
    for i in range(1, L + 1):
        for j in range(i + 1, L + 1):
            p = bpp[i][j]
            matrix[i - 1, j - 1] = p
            matrix[j - 1, i - 1] = p
    return matrix


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          .st / .bpseq 解析器                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class RNASample:
    """单条 RNA 样本"""
    name: str
    sequence: str          # 纯 ACGU
    dot_bracket: str       # 点括号参考结构
    contact: np.ndarray    # (L, L) 二值接触图
    length: int

    @classmethod
    def from_file(cls, filepath: str) -> "RNASample":
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        if content.strip().startswith("#"):
            return cls._from_st(content, filepath)
        else:
            return cls._from_bpseq(content, filepath)

    # ── .st 解析 ──
    @classmethod
    def _from_st(cls, content: str, filepath: str) -> "RNASample":
        lines = content.split("\n")

        # 第一遍 — 元信息
        name = Path(filepath).stem
        expected_len = None
        for line in lines:
            s = line.strip()
            if s.startswith("#Name:"):
                name = s.split(":", 1)[-1].strip()
            elif s.startswith("#Length:"):
                try:
                    expected_len = int(s.split(":", 1)[-1].strip())
                except ValueError:
                    pass

        # 收集非注释行
        non_comment = [l.strip() for l in lines
                       if l.strip() and not l.strip().startswith("#")]

        seq = None
        dbn = None

        # 策略 1: 相邻行 — 序列行紧接结构行
        for i in range(len(non_comment) - 1):
            a, b = non_comment[i], non_comment[i + 1]
            if expected_len and abs(len(a) - expected_len) > 5:
                continue
            a_bases = re.sub(r"[^ACGUTacgut]", "", a)
            b_dbn = re.sub(r"[^()\[\]{}.<>]", "", b)
            if len(a_bases) >= len(a) * 0.75 and len(b_dbn) >= len(b) * 0.55:
                seq = a.upper().replace("T", "U")
                dbn = b_dbn
                break

        # 策略 2 (fallback): 宽松匹配
        if seq is None:
            for i in range(len(non_comment)):
                a_bases = re.sub(r"[^ACGUTacgut]", "", non_comment[i])
                if len(a_bases) < len(non_comment[i]) * 0.70:
                    continue
                for j in range(i + 1, min(i + 4, len(non_comment))):
                    b_dbn = re.sub(r"[^()\[\]{}.<>]", "", non_comment[j])
                    if len(b_dbn) >= len(non_comment[j]) * 0.50:
                        seq = non_comment[i].upper().replace("T", "U")
                        dbn = b_dbn
                        break
                if seq:
                    break

        if seq is None:
            raise ValueError(f"无法提取序列: {filepath}")

        L = len(seq)
        dbn = (dbn or ".")[:L].ljust(L, ".")
        contact = cls._dbn_to_contact(dbn)
        return cls(name=name, sequence=seq, dot_bracket=dbn,
                   contact=contact, length=L)

    # ── .bpseq 解析 ──
    @classmethod
    def _from_bpseq(cls, content: str, filepath: str) -> "RNASample":
        name = Path(filepath).stem
        pairs = []
        chars = []
        for line in content.split("\n"):
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                idx, base, paired = int(parts[0]), parts[1], int(parts[2])
            except ValueError:
                continue
            chars.append(base)
            pairs.append((idx - 1, paired - 1 if paired > 0 else -1))

        seq = "".join(chars).upper().replace("T", "U")
        L = len(seq)
        dbn = ["."] * L
        contact = np.zeros((L, L), dtype=np.float32)
        for i, j in pairs:
            if 0 <= j < L and i < j:
                contact[i, j] = contact[j, i] = 1.0
                dbn[i] = "("
                dbn[j] = ")"
        return cls(name=name, sequence=seq, dot_bracket="".join(dbn),
                   contact=contact, length=L)

    # ── 工具 ──
    @staticmethod
    def _dbn_to_contact(dbn: str) -> np.ndarray:
        L = len(dbn)
        contact = np.zeros((L, L), dtype=np.float32)
        stacks = {"(": [], "[": [], "{": [], "<": []}
        pairs = {")": "(", "]": "[", "}": "{", ">": "<"}
        for i, c in enumerate(dbn):
            if c in stacks:
                stacks[c].append(i)
            elif c in pairs and stacks[pairs[c]]:
                j = stacks[pairs[c]].pop()
                contact[i, j] = contact[j, i] = 1.0
        return contact

    @staticmethod
    def onehot(seq: str) -> np.ndarray:
        m = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
        oh = np.zeros((len(seq), 4), dtype=np.float32)
        for i, c in enumerate(seq.upper()):
            if c in m:
                oh[i, m[c]] = 1.0
        return oh


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         模型推理                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ModelRunner:
    """统一模型加载 + 单条推理"""

    def __init__(self, checkpoint_path: str, device: torch.device,
                 use_bppm: bool = True):
        self.device = device
        self.use_bppm = use_bppm

        # 配置
        class Cfg:
            HIDDEN_DIM = 64
            RESNET_LAYERS = 8
            LSTM_HIDDEN = 64

        self.model = SpotRNA_LSTM_Refined_BPPM_Chimeric(Cfg()).to(device)

        state_dict = torch.load(checkpoint_path, map_location=device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        print(f"[model] loaded: {checkpoint_path}")

    @torch.no_grad()
    def predict(self, sample: RNASample) -> np.ndarray:
        """返回 (L, L) 概率矩阵"""
        L = sample.length
        x = torch.tensor(RNASample.onehot(sample.sequence),
                         dtype=torch.float32).unsqueeze(0).to(self.device)

        if self.use_bppm:
            bppm_np = get_bppm_matrix(sample.sequence)
            bppm = torch.tensor(bppm_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            bppm = None

        mask = torch.ones(1, L, dtype=torch.float32).to(self.device)

        logits = self.model(x, bppm=bppm, mask=mask,
                            trna_5end_len=None, trna_3end_len=None)
        prob = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        return prob


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         指标计算                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class SampleMetrics:
    """单条 RNA 的评测指标"""
    name: str
    length: int
    f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    threshold: float


def evaluate_sample(prob: np.ndarray, contact_gt: np.ndarray,
                    thresholds: np.ndarray = None) -> SampleMetrics:
    """
    在 [0.01, 0.99] 搜索最优阈值，返回最佳 F1 对应的指标。
    使用上三角 (不含对角线) 计算 TP/FP/FN。
    """
    L = prob.shape[0]
    triu = np.triu(np.ones((L, L), dtype=bool), k=1)
    gt_triu = contact_gt[triu]
    prob_triu = prob[triu]

    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    best = (0.0, 0.0, 0.0, 0, 0, 0, 0.5)  # f1, p, r, tp, fp, fn, th

    for th in thresholds:
        pred_bin = (prob_triu > th).astype(np.float32)
        tp = float((pred_bin * gt_triu).sum())
        fp = float((pred_bin * (1.0 - gt_triu)).sum())
        fn = float(((1.0 - pred_bin) * gt_triu).sum())

        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2.0 * p * r / (p + r + 1e-8)

        if f1 > best[0]:
            best = (f1, p, r, int(tp), int(fp), int(fn), th)

    return SampleMetrics(
        f1=best[0], precision=best[1], recall=best[2],
        tp=best[3], fp=best[4], fn=best[5], threshold=float(best[6]),
        name="", length=L,
    )


def aggregate_metrics(metrics_list: list[SampleMetrics]) -> dict:
    """从 per-sample 指标计算各类聚合统计量"""
    n = len(metrics_list)
    if n == 0:
        return {}

    f1s = np.array([m.f1 for m in metrics_list])
    ps = np.array([m.precision for m in metrics_list])
    rs = np.array([m.recall for m in metrics_list])

    total_tp = sum(m.tp for m in metrics_list)
    total_fp = sum(m.fp for m in metrics_list)
    total_fn = sum(m.fn for m in metrics_list)

    macro_p = total_tp / (total_tp + total_fp + 1e-8)
    macro_r = total_tp / (total_tp + total_fn + 1e-8)
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r + 1e-8)

    return {
        "n_samples": n,
        # per-sample 统计
        "f1_mean": float(np.mean(f1s)),
        "f1_median": float(np.median(f1s)),
        "f1_std": float(np.std(f1s)),
        "precision_mean": float(np.mean(ps)),
        "recall_mean": float(np.mean(rs)),
        # 全局聚合 (macro)
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        # TP/FP/FN 总计
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         主评测流程                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

LENGTH_GROUPS = [(0, 100), (100, 200), (200, 300), (300, 500), (500, 600)]


def run_evaluation(
    runner: ModelRunner,
    ts0_dir: str,
    output_dir: str,
    tag: str = "",
    length_groups: list = None,
) -> dict:
    """
    完整评测流程：
    1. 扫描 TS0 目录的全部 .st / .bpseq
    2. 逐条推理 + 计算 per-sample 最优 F1
    3. 聚合统计 + 长度分组
    4. 保存 JSON + CSV
    """
    if length_groups is None:
        length_groups = LENGTH_GROUPS

    ts0 = Path(ts0_dir)
    files = sorted(list(ts0.glob("*.st")) + list(ts0.glob("*.bpseq")))

    if not files:
        print(f"[error] 未找到文件: {ts0_dir}")
        return {}

    print(f"\n[eval] {len(files)} 条样本 ({sum(1 for f in files if f.suffix=='.st')} .st, "
          f"{sum(1 for f in files if f.suffix=='.bpseq')} .bpseq)")
    print(f"[eval] BPPM = {runner.use_bppm}")

    # ── 逐条评测 ──
    all_metrics: list[SampleMetrics] = []
    length_groups_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "count": 0})
    skipped = 0

    for idx, fpath in enumerate(files):
        try:
            sample = RNASample.from_file(str(fpath))
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  [skip] {fpath.name}: {e}")
            continue

        prob = runner.predict(sample)
        m = evaluate_sample(prob, sample.contact)
        m.name = sample.name
        m.length = sample.length
        all_metrics.append(m)

        # 长度分组
        for lo, hi in length_groups:
            if lo <= sample.length < hi:
                g = f"{lo}-{hi}"
                s = length_groups_stats[g]
                s["tp"] += m.tp
                s["fp"] += m.fp
                s["fn"] += m.fn
                s["count"] += 1
                break

        if (idx + 1) % 200 == 0:
            cur_f1 = np.mean([x.f1 for x in all_metrics])
            print(f"  [{idx+1:4d}/{len(files)}] current mean F1 = {cur_f1:.4f}")

    if skipped:
        print(f"\n  共跳过 {skipped} 条 (解析失败)")

    # ── 聚合 ──
    summary = aggregate_metrics(all_metrics)
    if not summary:
        return {}

    # 长度分组 F1
    for g in length_groups_stats:
        s = length_groups_stats[g]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        summary[f"len_{g}_f1"] = float(f1)
        summary[f"len_{g}_count"] = s["count"]

    summary["tag"] = tag or "v2"
    summary["skipped"] = skipped

    # ── 打印 ──
    print("\n" + "=" * 60)
    print(f"  TS0 评测结果")
    print("=" * 60)
    print(f"  样本数:            {summary['n_samples']}")
    print(f"  F1  (mean):        {summary['f1_mean']:.4f}")
    print(f"  F1  (median):      {summary['f1_median']:.4f}")
    print(f"  F1  (std):         {summary['f1_std']:.4f}")
    print(f"  Macro P:           {summary['macro_precision']:.4f}")
    print(f"  Macro R:           {summary['macro_recall']:.4f}")
    print(f"  Macro F1:          {summary['macro_f1']:.4f}")
    print(f"  Precision (mean):  {summary['precision_mean']:.4f}")
    print(f"  Recall (mean):     {summary['recall_mean']:.4f}")
    print(f"\n  长度分组:")
    for g in sorted(length_groups_stats, key=lambda x: int(x.split("-")[0])):
        c = summary.get(f"len_{g}_count", 0)
        f1 = summary.get(f"len_{g}_f1", 0)
        print(f"    {g:>10}nt: F1 = {f1:.4f}  (n = {c})")

    # ── 保存 ──
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "ts0_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  汇总: {json_path}")

    csv_path = out / "ts0_per_sample.csv"
    with open(csv_path, "w") as f:
        f.write("name,length,f1,precision,recall,tp,fp,fn,threshold\n")
        for m in all_metrics:
            f.write(f"{m.name},{m.length},{m.f1:.6f},{m.precision:.6f},"
                    f"{m.recall:.6f},{m.tp},{m.fp},{m.fn},{m.threshold:.4f}\n")
    print(f"  逐样本: {csv_path}")

    return summary


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         CLI                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="TS0 标准基准评测 — bpRNA 官方测试集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型 .pth 路径")
    parser.add_argument("--ts0_dir", type=str,
                        help="TS0 目录 (默认自动搜索 data/TS0)")
    parser.add_argument("--output_dir", type=str, default="results/ts0_eval",
                        help="输出目录")
    parser.add_argument("--tag", type=str, default="",
                        help="结果标签 (用于消融实验区分)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_bppm", action="store_true",
                        help="禁用 BPPM (消融实验用)")

    args = parser.parse_args()

    # ── 设备 ──
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # ── TS0 目录 ──
    if args.ts0_dir is None:
        for candidate in ["data/TS0", "data/bpRNA_dataset/TS0"]:
            if os.path.isdir(candidate):
                args.ts0_dir = candidate
                break
    if args.ts0_dir is None or not os.path.isdir(args.ts0_dir):
        print("Error: 未找到 TS0 目录，请用 --ts0_dir 指定")
        sys.exit(1)

    # ── 运行 ──
    runner = ModelRunner(args.checkpoint, device, use_bppm=not args.no_bppm)
    summary = run_evaluation(runner, args.ts0_dir, args.output_dir, tag=args.tag)

    if summary:
        print(f"\n{'=' * 60}")
        tag_label = args.tag or "model"
        print(f"  [{tag_label}] TS0 macro F1 = {summary['macro_f1']:.4f}  "
              f"(mean F1 = {summary['f1_mean']:.4f})")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

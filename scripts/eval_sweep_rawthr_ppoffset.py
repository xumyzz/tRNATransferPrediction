#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended evaluation sweep for:
1) RAW threshold sweep (raw_thr)
2) PP offset sweep (pp_offset)
3) Joint sweep grid (raw_thr x pp_offset)
4) GT one-to-one pairing check (degree statistics)
5) Save Top-K improved/degraded sample indices for each (raw_thr, pp_offset) cell

Outputs:
- Macro avg (per-sample) F1/P/R for RAW and PP
- improved/degraded fractions relative to RAW (at the same raw_thr)
- degree stats of GT contact maps: how often a nucleotide pairs with >1 partner

Example:
  PYTHONPATH=. python scripts/eval_sweep_rawthr_ppoffset.py \
    --data_dir /root/autodl-tmp/newPredicProject/dbnFiles \
    --ckpt /root/autodl-tmp/newPredicProject/runs/finetune_seed42_offset0p35/model_best.pth \
    --max_len 100 \
    --num_samples 2000 \
    --min_loop 3 \
    --raw_thrs 0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90 \
    --pp_offsets 0.35,0.40,0.45,0.50,0.55,0.60,0.65 \
    --improve_delta 0.05 \
    --degrade_delta 0.05 \
    --topk 25 \
    --out_dir /root/autodl-tmp/newPredicProject/sweep_reports_len100 \
    --device cuda

Reproducibility:
- Use --indices_file to lock the evaluation subset across runs.
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.dataset import MultiFileDatasetUpgrade
from src.model import SpotRNA_LSTM_Refined
from src.config import Config
from src.metrics import calculate_f1_postprocess_ufold


# ----------------- helpers -----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_mask_from_onehot(seqs_1d: torch.Tensor) -> torch.Tensor:
    return (seqs_1d.abs().sum(dim=-1) > 0)


def unpack_dataset_item(item):
    if isinstance(item, dict):
        return item["seqs"], item["labels"], item.get("masks", None)
    if isinstance(item, (tuple, list)):
        if len(item) == 2:
            return item[0], item[1], None
        if len(item) == 3:
            return item[0], item[1], item[2]
    raise ValueError(f"Unsupported dataset item type/len: {type(item)}")


def parse_floats(s: str) -> List[float]:
    s = s.replace(",", " ").strip()
    return [float(x) for x in s.split() if x]


def load_indices(indices_file: Optional[str], n_total: int) -> Optional[List[int]]:
    if not indices_file:
        return None
    if not os.path.exists(indices_file):
        return None
    arr = np.load(indices_file)
    idxs = [int(x) for x in arr.tolist()]
    for i in idxs:
        if i < 0 or i >= n_total:
            raise ValueError(f"indices_file contains out-of-range index {i} (n_total={n_total})")
    return idxs


def save_indices(indices_file: str, idxs: List[int]):
    os.makedirs(os.path.dirname(indices_file) or ".", exist_ok=True)
    np.save(indices_file, np.array(idxs, dtype=np.int64))


def prf_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return f1, p, r


@dataclass
class Agg:
    n: int = 0
    sum_f1: float = 0.0
    sum_p: float = 0.0
    sum_r: float = 0.0

    def add(self, f1: float, p: float, r: float):
        self.n += 1
        self.sum_f1 += float(f1)
        self.sum_p += float(p)
        self.sum_r += float(r)

    def mean(self) -> Tuple[float, float, float]:
        if self.n == 0:
            return 0.0, 0.0, 0.0
        return self.sum_f1 / self.n, self.sum_p / self.n, self.sum_r / self.n


@dataclass
class TopKDelta:
    # store (delta, idx) and keep top-k for improved and degraded
    k: int
    improved: List[Tuple[float, int]]
    degraded: List[Tuple[float, int]]

    def __init__(self, k: int):
        self.k = k
        self.improved = []
        self.degraded = []

    def add(self, idx: int, raw_f1: float, pp_f1: float):
        d = float(pp_f1 - raw_f1)
        # improved: d positive
        if d > 0:
            self.improved.append((d, idx))
            self.improved.sort(reverse=True)
            if len(self.improved) > self.k:
                self.improved = self.improved[: self.k]
        # degraded: d negative, keep most negative
        if d < 0:
            self.degraded.append((d, idx))
            self.degraded.sort()  # ascending, most negative first
            if len(self.degraded) > self.k:
                self.degraded = self.degraded[: self.k]


@torch.no_grad()
def raw_counts_upper(probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, thr: float) -> Tuple[int, int, int]:
    """
    probs: (L,L) in [0,1]
    labels: (L,L) 0/1
    mask: (L,) bool
    upper triangle, exclude diagonal
    """
    L = labels.size(0)
    m = mask.bool()
    m2 = (m[:, None] & m[None, :])

    diag = torch.eye(L, device=labels.device).bool()
    m2 = m2 & (~diag)

    triu = torch.triu(torch.ones((L, L), dtype=torch.bool, device=labels.device), diagonal=1)
    valid = m2 & triu

    y_true = (labels > 0.5) & valid
    y_pred = (probs >= thr) & valid

    tp = torch.logical_and(y_pred, y_true).sum().item()
    fp = torch.logical_and(y_pred, ~y_true).sum().item()
    fn = torch.logical_and(~y_pred, y_true).sum().item()
    return int(tp), int(fp), int(fn)


@torch.no_grad()
def raw_metrics_from_probs(probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, thr: float) -> Tuple[float, float, float]:
    tp, fp, fn = raw_counts_upper(probs, labels, mask, thr)
    return prf_from_counts(tp, fp, fn)


@torch.no_grad()
def gt_degree_stats(labels: torch.Tensor, mask: torch.Tensor) -> Tuple[int, int, int]:
    """
    labels: (L,L) 0/1
    mask: (L,) bool
    returns:
      n_valid_positions, n_positions_deg_gt1, max_deg
    Degree counts how many partners each i has in GT (full matrix, excluding diagonal).
    """
    L = labels.size(0)
    m = mask.bool()
    # valid positions are those with mask True
    valid_idx = torch.where(m)[0]
    if valid_idx.numel() == 0:
        return 0, 0, 0

    # restrict to valid region and exclude diagonal
    lab = (labels > 0.5).clone()
    lab.fill_diagonal_(0)

    # degree per position counts across all j in valid region
    deg = []
    for i in valid_idx.tolist():
        # count partners only among valid positions
        di = lab[i, valid_idx].sum().item()
        deg.append(int(di))

    n_valid = len(deg)
    n_gt1 = sum(1 for d in deg if d > 1)
    max_deg = max(deg) if deg else 0
    return n_valid, n_gt1, max_deg


def write_list(path: str, rows: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.rstrip("\n") + "\n")


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_len", type=int, default=600)
    ap.add_argument("--device", type=str, default=None)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_samples", type=int, default=2000)

    ap.add_argument("--min_loop", type=int, default=3)
    ap.add_argument("--raw_thrs", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--pp_offsets", type=str, default="0.35,0.4,0.45,0.5,0.55,0.6,0.65")

    ap.add_argument("--improve_delta", type=float, default=0.05)
    ap.add_argument("--degrade_delta", type=float, default=0.05)

    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--out_dir", type=str, default="sweep_reports")
    ap.add_argument("--indices_file", type=str, default=None,
                    help="If provided, load/save sampled indices as .npy for reproducibility.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)

    raw_thrs = parse_floats(args.raw_thrs)
    pp_offsets = parse_floats(args.pp_offsets)
    if not raw_thrs or not pp_offsets:
        raise ValueError("raw_thrs and pp_offsets must be non-empty")

    ds = MultiFileDatasetUpgrade(data_dir_or_file=args.data_dir, max_len=args.max_len)
    n_total = len(ds)
    if n_total == 0:
        raise RuntimeError("Dataset is empty")

    # indices
    idxs = load_indices(args.indices_file, n_total)
    if idxs is None:
        idxs = list(range(n_total))
        random.shuffle(idxs)
        idxs = idxs[: min(args.num_samples, n_total)]
        if args.indices_file:
            save_indices(args.indices_file, idxs)
            print(f"Saved sampled indices -> {args.indices_file} (n={len(idxs)})")
    else:
        print(f"Loaded fixed indices from {args.indices_file} (n={len(idxs)})")

    # model
    cfg = Config
    cfg.MAX_LEN = args.max_len
    model = SpotRNA_LSTM_Refined(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    msg = model.load_state_dict(state, strict=False)
    print(f"Loaded ckpt: {args.ckpt}")
    if hasattr(msg, "missing_keys"):
        print(f"Missing keys: {len(msg.missing_keys)}  Unexpected keys: {len(msg.unexpected_keys)}")
    model.eval()

    # accumulators
    # raw macro metrics for each raw_thr
    raw_aggs: Dict[float, Agg] = {t: Agg() for t in raw_thrs}

    # pp macro metrics and improved/degraded counts for each (raw_thr, pp_offset)
    pp_aggs: Dict[Tuple[float, float], Agg] = {(t, o): Agg() for t in raw_thrs for o in pp_offsets}
    improved_cnt: Dict[Tuple[float, float], int] = {(t, o): 0 for t in raw_thrs for o in pp_offsets}
    degraded_cnt: Dict[Tuple[float, float], int] = {(t, o): 0 for t in raw_thrs for o in pp_offsets}
    topk_delta: Dict[Tuple[float, float], TopKDelta] = {(t, o): TopKDelta(args.topk) for t in raw_thrs for o in pp_offsets}

    # GT degree stats
    total_valid_positions = 0
    total_deg_gt1_positions = 0
    max_deg_overall = 0
    n_sequences_degree_checked = 0

    for it, idx in enumerate(idxs, start=1):
        seqs, labels, masks = unpack_dataset_item(ds[idx])

        if not isinstance(seqs, torch.Tensor):
            seqs = torch.tensor(seqs)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        if masks is not None and not isinstance(masks, torch.Tensor):
            masks = torch.tensor(masks)

        if seqs.dim() != 2:
            raise ValueError(f"Expected seqs dim=2 (L,4). Got {seqs.shape}")

        if masks is None:
            mask_1d = infer_mask_from_onehot(seqs)
        else:
            mask_1d = masks.bool() if masks.dim() == 1 else masks[0].bool()

        L = int(mask_1d.sum().item())
        if L <= 1:
            continue

        seqs_b = seqs.unsqueeze(0).to(device)
        labels = labels.to(device)
        mask_1d = mask_1d.to(device)

        with torch.no_grad():
            logits = model(seqs_b)[0]  # (MAX_LEN, MAX_LEN)

        logits_v = logits[:L, :L]
        labels_v = labels[:L, :L]
        seqs_v = seqs[:L, :].to(device)
        mask_v = mask_1d[:L]

        # precompute probs once for raw sweeps
        probs_v = torch.sigmoid(logits_v)

        # GT degree stats once per sample
        n_valid, n_gt1, mx = gt_degree_stats(labels_v, mask_v)
        if n_valid > 0:
            total_valid_positions += n_valid
            total_deg_gt1_positions += n_gt1
            max_deg_overall = max(max_deg_overall, mx)
            n_sequences_degree_checked += 1

        # RAW for each threshold
        raw_f1_by_thr: Dict[float, float] = {}
        for thr in raw_thrs:
            raw_f1, raw_p, raw_r = raw_metrics_from_probs(probs_v, labels_v, mask_v, thr=thr)
            raw_aggs[thr].add(raw_f1, raw_p, raw_r)
            raw_f1_by_thr[thr] = raw_f1

        # PP for each offset, compared against RAW at each thr
        for off in pp_offsets:
            pp_f1, pp_p, pp_r = calculate_f1_postprocess_ufold(
                logits=logits_v.unsqueeze(0),
                labels=labels_v.unsqueeze(0),
                seqs=seqs_v.unsqueeze(0),
                masks=mask_v.unsqueeze(0),
                offset=float(off),
                min_loop=int(args.min_loop),
            )

            for thr in raw_thrs:
                key = (thr, off)
                pp_aggs[key].add(pp_f1, pp_p, pp_r)

                raw_f1 = raw_f1_by_thr[thr]
                if (pp_f1 - raw_f1) >= args.improve_delta:
                    improved_cnt[key] += 1
                if (raw_f1 - pp_f1) >= args.degrade_delta:
                    degraded_cnt[key] += 1
                topk_delta[key].add(idx=idx, raw_f1=raw_f1, pp_f1=pp_f1)

        if it % 200 == 0:
            print(f"Processed {it}/{len(idxs)} samples...")

    # ----------------- report -----------------
    # degree stats
    deg_gt1_frac = (total_deg_gt1_positions / max(1, total_valid_positions))
    print("\n=== GT one-to-one check ===")
    print(f"Sequences checked: {n_sequences_degree_checked}")
    print(f"Total valid positions: {total_valid_positions}")
    print(f"Positions with GT degree>1: {total_deg_gt1_positions} ({deg_gt1_frac:.3%})")
    print(f"Max GT degree observed: {max_deg_overall}")
    write_list(
        os.path.join(args.out_dir, "gt_degree_stats.txt"),
        [
            f"Sequences checked: {n_sequences_degree_checked}",
            f"Total valid positions: {total_valid_positions}",
            f"Positions with GT degree>1: {total_deg_gt1_positions} ({deg_gt1_frac:.6f})",
            f"Max GT degree observed: {max_deg_overall}",
        ],
    )

    # RAW best thr
    raw_rows = []
    best_thr = None
    best_raw_f1 = -1.0
    for thr in raw_thrs:
        f1, p, r = raw_aggs[thr].mean()
        raw_rows.append(f"RAW thr={thr:.3f}: F1={f1:.6f} P={p:.6f} R={r:.6f}")
        if f1 > best_raw_f1:
            best_raw_f1 = f1
            best_thr = thr

    print("\n=== RAW sweep (macro avg) ===")
    for line in raw_rows:
        print(line)
    print(f"RAW best thr={best_thr:.3f} with F1={best_raw_f1:.6f}")
    write_list(os.path.join(args.out_dir, "raw_sweep.txt"), raw_rows + [f"BEST thr={best_thr:.3f} F1={best_raw_f1:.6f}"])

    # PP grid summary table
    n_eval = next(iter(raw_aggs.values())).n
    grid_lines = []
    grid_lines.append(f"Evaluated samples (macro): {n_eval}")
    grid_lines.append(f"min_loop={args.min_loop} improve_delta={args.improve_delta} degrade_delta={args.degrade_delta}")
    grid_lines.append("Columns: raw_thr, pp_offset, raw_F1, pp_F1, pp_P, pp_R, improved_frac, degraded_frac")

    best_pp_vs_raw = None  # best (thr, off) by PP F1
    best_pp_f1 = -1.0

    for thr in raw_thrs:
        raw_f1, _, _ = raw_aggs[thr].mean()
        for off in pp_offsets:
            key = (thr, off)
            pp_f1, pp_p, pp_r = pp_aggs[key].mean()
            improved_frac = improved_cnt[key] / max(1, n_eval)
            degraded_frac = degraded_cnt[key] / max(1, n_eval)

            grid_lines.append(
                f"raw_thr={thr:.3f} pp_offset={off:.3f} | "
                f"rawF1={raw_f1:.6f} ppF1={pp_f1:.6f} ppP={pp_p:.6f} ppR={pp_r:.6f} | "
                f"improved={improved_frac:.3%} degraded={degraded_frac:.3%}"
            )

            if pp_f1 > best_pp_f1:
                best_pp_f1 = pp_f1
                best_pp_vs_raw = (thr, off)

    print("\n=== PP grid (macro avg) ===")
    print(f"Best PP cell by PP-F1: raw_thr={best_pp_vs_raw[0]:.3f}, pp_offset={best_pp_vs_raw[1]:.3f}, ppF1={best_pp_f1:.6f}")
    write_list(os.path.join(args.out_dir, "pp_grid.txt"), grid_lines + [
        f"BEST_PP raw_thr={best_pp_vs_raw[0]:.3f} pp_offset={best_pp_vs_raw[1]:.3f} ppF1={best_pp_f1:.6f}"
    ])

    # Save Top-K examples for each cell
    # Each file contains two sections: improved (largest +delta) and degraded (most negative delta)
    for thr in raw_thrs:
        for off in pp_offsets:
            key = (thr, off)
            t = topk_delta[key]

            lines = []
            lines.append(f"Cell raw_thr={thr:.3f} pp_offset={off:.3f} min_loop={args.min_loop}")
            lines.append(f"TopK={args.topk} | delta = pp_f1 - raw_f1")
            lines.append("")
            lines.append("[IMPROVED] (largest positive delta)")
            for d, idx in t.improved:
                lines.append(f"delta={d:+.6f} idx={idx}")
            lines.append("")
            lines.append("[DEGRADED] (most negative delta)")
            for d, idx in t.degraded:
                lines.append(f"delta={d:+.6f} idx={idx}")

            out_path = os.path.join(args.out_dir, f"topk_rawthr{thr:.2f}_ppoff{off:.2f}.txt")
            write_list(out_path, lines)

    print(f"\nWrote reports to: {args.out_dir}")
    print("Key files:")
    print(f"  - {os.path.join(args.out_dir, 'gt_degree_stats.txt')}")
    print(f"  - {os.path.join(args.out_dir, 'raw_sweep.txt')}")
    print(f"  - {os.path.join(args.out_dir, 'pp_grid.txt')}")
    print("  - topk_rawthr*_ppoff*.txt (per cell)")


if __name__ == "__main__":
    main()
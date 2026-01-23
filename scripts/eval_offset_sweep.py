#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep UFold-style postprocess offset and compare PP vs RAW on a fixed dataset split.

What you get per offset:
- RAW (thr=0.5 by default): mean F1/P/R across samples
- PP (offset, min_loop): mean F1/P/R across samples
- Fraction of samples where PP is significantly better/worse than RAW:
    * pp_f1 - raw_f1 >= improve_delta  -> "improved"
    * raw_f1 - pp_f1 >= degrade_delta  -> "degraded"

Notes:
- "mean across samples" here is per-sample F1 averaged (macro). This matches the intuition
  of your visualization script (each idx has its own F1).
- We compute RAW metrics with upper-triangle masking (no diagonal, no double counting).
- PP metrics use your existing calculate_f1_postprocess_ufold for correctness.

Example:
  PYTHONPATH=. python scripts/eval_offset_sweep.py \
    --data_dir /root/autodl-tmp/newPredicProject/dbnFiles \
    --ckpt /root/autodl-tmp/newPredicProject/runs/finetune_seed42_offset0p35/model_best.pth \
    --max_len 100 \
    --num_samples 2000 \
    --raw_thr 0.5 \
    --min_loop 4 \
    --offsets 0.35,0.40,0.45,0.50,0.55,0.60,0.65 \
    --improve_delta 0.05 \
    --degrade_delta 0.05 \
    --device cuda

Tip:
- Use --indices_file to evaluate exactly the same samples every time.
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch

from src.dataset import MultiFileDatasetUpgrade
from src.model import SpotRNA_LSTM_Refined
from src.config import Config
from src.metrics import calculate_f1_postprocess_ufold


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


def prf_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return f1, p, r


@torch.no_grad()
def raw_f1_single(logits_v: torch.Tensor, labels_v: torch.Tensor, mask_v: torch.Tensor, thr: float) -> Tuple[float, float, float]:
    """
    logits_v: (L,L)
    labels_v: (L,L) 0/1
    mask_v:   (L,) bool
    Computes RAW metrics using upper triangle, excluding diagonal.
    """
    probs = torch.sigmoid(logits_v)

    L = labels_v.size(0)
    m = mask_v.bool()
    m2 = (m[:, None] & m[None, :])

    diag = torch.eye(L, device=labels_v.device).bool()
    m2 = m2 & (~diag)

    triu = torch.triu(torch.ones((L, L), dtype=torch.bool, device=labels_v.device), diagonal=1)
    m2 = m2 & triu

    y_true = (labels_v > 0.5) & m2
    y_pred = (probs >= thr) & m2

    tp = torch.logical_and(y_pred, y_true).sum().item()
    fp = torch.logical_and(y_pred, ~y_true).sum().item()
    fn = torch.logical_and(~y_pred, y_true).sum().item()

    return prf_from_counts(int(tp), int(fp), int(fn))


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


def parse_offsets(s: str) -> List[float]:
    # allow "0.35,0.4,0.45" or "0.35 0.4 0.45"
    s = s.replace(",", " ").strip()
    return [float(x) for x in s.split() if x]


def load_indices(indices_file: Optional[str], n_total: int) -> Optional[List[int]]:
    if not indices_file:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_len", type=int, default=600)
    ap.add_argument("--device", type=str, default=None)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_samples", type=int, default=1000, help="How many random samples to evaluate (macro average).")

    ap.add_argument("--raw_thr", type=float, default=0.5)

    ap.add_argument("--min_loop", type=int, default=4)
    ap.add_argument("--offsets", type=str, default="0.35 0.4 0.45 0.5 0.55 0.6 0.65")

    ap.add_argument("--improve_delta", type=float, default=0.05)
    ap.add_argument("--degrade_delta", type=float, default=0.05)

    ap.add_argument("--indices_file", type=str, default=None,
                    help="If provided, load indices from .npy and evaluate exactly those samples. "
                         "If file doesn't exist, it will be created using the sampled indices.")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ds = MultiFileDatasetUpgrade(data_dir_or_file=args.data_dir, max_len=args.max_len)
    n_total = len(ds)
    if n_total == 0:
        raise RuntimeError("Dataset is empty")

    # choose indices
    fixed = load_indices(args.indices_file, n_total)
    if fixed is not None:
        idxs = fixed
        print(f"Loaded {len(idxs)} fixed indices from {args.indices_file}")
    else:
        idxs = list(range(n_total))
        random.shuffle(idxs)
        idxs = idxs[: min(args.num_samples, n_total)]
        if args.indices_file:
            # save for reproducibility
            if not os.path.exists(args.indices_file):
                save_indices(args.indices_file, idxs)
                print(f"Saved sampled indices to {args.indices_file} (n={len(idxs)})")

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

    offsets = parse_offsets(args.offsets)
    if len(offsets) == 0:
        raise ValueError("--offsets is empty")

    # accumulators
    raw_agg = Agg()
    pp_aggs = {off: Agg() for off in offsets}

    improved_cnt = {off: 0 for off in offsets}
    degraded_cnt = {off: 0 for off in offsets}

    # main loop
    for t, idx in enumerate(idxs, start=1):
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
            logits = model(seqs_b)[0]  # (MAX_LEN, MAX_LEN) padded

        # crop to valid length
        logits_v = logits[:L, :L]
        labels_v = labels[:L, :L]
        seqs_v = seqs[:L, :].to(device)
        mask_v = mask_1d[:L]

        # RAW per-sample
        raw_f1, raw_p, raw_r = raw_f1_single(logits_v, labels_v, mask_v, thr=args.raw_thr)
        raw_agg.add(raw_f1, raw_p, raw_r)

        # PP per-sample for each offset
        for off in offsets:
            pp_f1, pp_p, pp_r = calculate_f1_postprocess_ufold(
                logits=logits_v.unsqueeze(0),
                labels=labels_v.unsqueeze(0),
                seqs=seqs_v.unsqueeze(0),
                masks=mask_v.unsqueeze(0),
                offset=float(off),
                min_loop=int(args.min_loop),
            )
            pp_aggs[off].add(pp_f1, pp_p, pp_r)

            if (pp_f1 - raw_f1) >= args.improve_delta:
                improved_cnt[off] += 1
            if (raw_f1 - pp_f1) >= args.degrade_delta:
                degraded_cnt[off] += 1

        if t % 200 == 0:
            print(f"Processed {t}/{len(idxs)} samples...")

    # report
    n_eval = raw_agg.n
    raw_mean = raw_agg.mean()

    print("\n=== Summary (macro avg over samples) ===")
    print(f"Evaluated samples: {n_eval}")
    print(f"RAW thr={args.raw_thr}:  F1={raw_mean[0]:.4f}  P={raw_mean[1]:.4f}  R={raw_mean[2]:.4f}")

    print("\n=== Postprocess sweep ===")
    for off in offsets:
        pp_mean = pp_aggs[off].mean()
        improved_frac = improved_cnt[off] / max(1, n_eval)
        degraded_frac = degraded_cnt[off] / max(1, n_eval)
        print(
            f"PP offset={off:.3f} min_loop={args.min_loop}: "
            f"F1={pp_mean[0]:.4f} P={pp_mean[1]:.4f} R={pp_mean[2]:.4f} | "
            f"improved>={args.improve_delta:.2f}: {improved_frac:.3%}  "
            f"degraded>={args.degrade_delta:.2f}: {degraded_frac:.3%}"
        )


if __name__ == "__main__":
    main()
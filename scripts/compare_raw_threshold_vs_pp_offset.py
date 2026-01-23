#!/usr/bin/env python3
"""
Compare raw-threshold decoding vs UFold-style postprocess-offset decoding on a validation set.

It will:
- load a checkpoint
- build the same val split (cluster split if --clstr_path provided)
- run inference on val
- compute F1/P/R for:
  (A) RAW: sigmoid(logits) >= threshold
  (B) PP : UFold postprocess + matching with offset

Then it prints two tables and saves a PNG plot with two curves:
  raw_f1(threshold) and pp_f1(offset)

Usage example:
  PYTHONPATH=. python scripts/compare_raw_threshold_vs_pp_offset.py \
    --data_dir /root/autodl-tmp/newPredicProject/dbnFiles \
    --max_len 600 \
    --clstr_path /root/autodl-tmp/newPredicProject/bprna1m_cdhit90_len600.clstr \
    --train_frac 0.98 --val_frac 0.01 --seed 42 \
    --ckpt /root/autodl-tmp/newPredicProject/model_best.pth \
    --batch_size 2 --device cuda \
    --min_loop 4 \
    --raw_min 0.10 --raw_max 0.90 --raw_step 0.05 \
    --pp_min 0.10 --pp_max 0.80 --pp_step 0.05 \
    --out_png /root/autodl-tmp/newPredicProject/raw_vs_pp.png \
    --max_val_batches 50

Notes:
- Postprocess uses src.metrics.calculate_f1_postprocess_ufold (same as your training val).
- RAW metric uses upper triangle only, same masking rules (no matching/canonical filtering).
- For speed, use --max_val_batches to subsample.
"""

import argparse
import os
import random
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# project imports
from src.config import Config
from src.dataset import MultiFileDatasetUpgrade, collate_pad
from src.model import SpotRNA_LSTM_Refined
from src.metrics import calculate_f1_postprocess_ufold
from scripts.cluster_utils import parse_cd_hit_clusters


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def frange(start: float, stop: float, step: float) -> List[float]:
    xs = []
    x = start
    while x <= stop + 1e-12:
        xs.append(round(x, 10))
        x += step
    return xs


def create_cluster_split(dataset, clstr_path, train_frac, val_frac, seed=42):
    random.seed(seed)
    clusters = parse_cd_hit_clusters(clstr_path)
    name_to_idx = {name: idx for idx, name in enumerate(dataset.names)}

    cluster_indices = []
    for cluster in clusters:
        indices = [name_to_idx[name] for name in cluster if name in name_to_idx]
        if indices:
            cluster_indices.append(indices)

    random.shuffle(cluster_indices)

    n_clusters = len(cluster_indices)
    n_train = int(n_clusters * train_frac)
    n_val = int(n_clusters * val_frac)

    val_clusters = cluster_indices[n_train:n_train + n_val]
    val_indices = [idx for c in val_clusters for idx in c]
    return val_indices


@torch.no_grad()
def raw_batch_counts(logits, labels, masks, threshold: float) -> Tuple[int, int, int]:
    """
    Compute (TP, FP, FN) for a batch using raw threshold on sigmoid(logits).
    Uses:
    - valid positions from masks
    - upper triangle only
    - ignore diagonal
    """
    probs = torch.sigmoid(logits)
    B, L, _ = probs.shape

    # build mask for each sample
    tp = fp = fn = 0
    for b in range(B):
        m = masks[b].bool()
        m2 = (m[:, None] & m[None, :])

        triu = torch.triu(torch.ones((L, L), dtype=torch.bool, device=logits.device), diagonal=1)
        m2 = m2 & triu  # includes diag excluded

        y_true = labels[b].bool() & m2
        y_pred = (probs[b] >= threshold) & m2

        tp += torch.logical_and(y_pred, y_true).sum().item()
        fp += torch.logical_and(y_pred, ~y_true).sum().item()
        fn += torch.logical_and(~y_pred, y_true).sum().item()

    return int(tp), int(fp), int(fn)


def prf_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return f1, p, r


@torch.no_grad()
def eval_curves(
    model,
    loader,
    device,
    raw_thresholds: List[float],
    pp_offsets: List[float],
    min_loop: int,
    max_val_batches: int | None,
):
    # RAW: accumulate global TP/FP/FN per threshold (micro)
    raw_counts: Dict[float, Dict[str, int]] = {
        t: {"tp": 0, "fp": 0, "fn": 0} for t in raw_thresholds
    }

    # PP: your existing metric returns per-batch averages; we will average across batches to match training logs
    pp_sum: Dict[float, Dict[str, float]] = {
        o: {"f1": 0.0, "p": 0.0, "r": 0.0} for o in pp_offsets
    }

    n_batches = 0
    model.eval()

    for seqs, labels, masks in loader:
        n_batches += 1
        seqs = seqs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        logits = model(seqs, mask=masks)

        # RAW curve
        for t in raw_thresholds:
            tp, fp, fn = raw_batch_counts(logits, labels, masks, threshold=t)
            raw_counts[t]["tp"] += tp
            raw_counts[t]["fp"] += fp
            raw_counts[t]["fn"] += fn

        # PP curve
        for o in pp_offsets:
            f1, p, r = calculate_f1_postprocess_ufold(
                logits=logits,
                labels=labels,
                seqs=seqs,
                masks=masks,
                offset=o,
                min_loop=min_loop,
            )
            pp_sum[o]["f1"] += f1
            pp_sum[o]["p"] += p
            pp_sum[o]["r"] += r

        if max_val_batches is not None and n_batches >= max_val_batches:
            break

    # finalize RAW
    raw_metrics: Dict[float, Tuple[float, float, float]] = {}
    for t in raw_thresholds:
        c = raw_counts[t]
        raw_metrics[t] = prf_from_counts(c["tp"], c["fp"], c["fn"])

    # finalize PP (batch-average)
    pp_metrics: Dict[float, Tuple[float, float, float]] = {}
    for o in pp_offsets:
        denom = max(n_batches, 1)
        pp_metrics[o] = (
            pp_sum[o]["f1"] / denom,
            pp_sum[o]["p"] / denom,
            pp_sum[o]["r"] / denom,
        )

    return raw_metrics, pp_metrics, n_batches


def plot_curves(raw_metrics, pp_metrics, out_png):
    raw_x = sorted(raw_metrics.keys())
    raw_f1 = [raw_metrics[t][0] for t in raw_x]

    pp_x = sorted(pp_metrics.keys())
    pp_f1 = [pp_metrics[o][0] for o in pp_x]

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(raw_x, raw_f1, marker="o", label="RAW: sigmoid >= threshold")
    plt.plot(pp_x, pp_f1, marker="o", label="PP : UFold matching offset")
    plt.xlabel("threshold / offset")
    plt.ylabel("F1")
    plt.title("F1 curve: RAW threshold vs Postprocess offset")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def print_table(title, xs, metrics):
    print(f"\n{title}")
    header = f"{'x':>8} | {'F1':>7} | {'P':>7} | {'R':>7}"
    print(header)
    print("-" * len(header))
    for x in xs:
        f1, p, r = metrics[x]
        print(f"{x:8.2f} | {f1:7.4f} | {p:7.4f} | {r:7.4f}")


def select_best(xs, metrics):
    best_x = None
    best_f1 = -1.0
    for x in xs:
        f1 = metrics[x][0]
        if f1 > best_f1:
            best_f1 = f1
            best_x = x
    return best_x, metrics[best_x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_len", type=int, default=600)

    ap.add_argument("--clstr_path", type=str, default=None)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--resnet_layers", type=int, default=8)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--lstm_hidden", type=int, default=64)

    ap.add_argument("--min_loop", type=int, default=4)

    ap.add_argument("--raw_min", type=float, default=0.10)
    ap.add_argument("--raw_max", type=float, default=0.90)
    ap.add_argument("--raw_step", type=float, default=0.05)

    ap.add_argument("--pp_min", type=float, default=0.10)
    ap.add_argument("--pp_max", type=float, default=0.80)
    ap.add_argument("--pp_step", type=float, default=0.05)

    ap.add_argument("--out_png", type=str, default="raw_vs_pp.png")
    ap.add_argument("--max_val_batches", type=int, default=None, help="Subsample for speed (e.g., 50)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    raw_thresholds = frange(args.raw_min, args.raw_max, args.raw_step)
    pp_offsets = frange(args.pp_min, args.pp_max, args.pp_step)

    print("RAW thresholds:", raw_thresholds)
    print("PP offsets:", pp_offsets)

    ds = MultiFileDatasetUpgrade(args.data_dir, max_len=args.max_len)
    if len(ds) == 0:
        raise RuntimeError("Dataset empty. Check --data_dir")

    if args.clstr_path:
        val_idx = create_cluster_split(ds, args.clstr_path, args.train_frac, args.val_frac, args.seed)
        val_ds = Subset(ds, val_idx)
        print(f"Using cluster split: val sequences={len(val_ds)}")
    else:
        # random split fallback
        n = len(ds)
        n_train = int(args.train_frac * n)
        n_val = int(args.val_frac * n)
        g = torch.Generator().manual_seed(args.seed)
        _, val_ds, _ = torch.utils.data.random_split(ds, [n_train, n_val, n - n_train - n_val], generator=g)
        print(f"Using random split: val sequences={len(val_ds)}")

    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad, num_workers=0)

    cfg = Config()
    cfg.RESNET_LAYERS = args.resnet_layers
    cfg.HIDDEN_DIM = args.hidden_dim
    cfg.LSTM_HIDDEN = args.lstm_hidden
    cfg.DEVICE = device
    model = SpotRNA_LSTM_Refined(cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    raw_metrics, pp_metrics, n_batches = eval_curves(
        model=model,
        loader=loader,
        device=device,
        raw_thresholds=raw_thresholds,
        pp_offsets=pp_offsets,
        min_loop=args.min_loop,
        max_val_batches=args.max_val_batches,
    )
    print(f"Evaluated {n_batches} batches")

    print_table("RAW curve (micro counts across val)", raw_thresholds, raw_metrics)
    best_t, (best_f1, best_p, best_r) = select_best(raw_thresholds, raw_metrics)
    print(f"\nRAW best: thr={best_t:.2f}  F1={best_f1:.4f}  P={best_p:.4f}  R={best_r:.4f}")

    print_table("PP curve (batch-avg like your training logs)", pp_offsets, pp_metrics)
    best_o, (best_f1, best_p, best_r) = select_best(pp_offsets, pp_metrics)
    print(f"\nPP best: offset={best_o:.2f}  F1={best_f1:.4f}  P={best_p:.4f}  R={best_r:.4f}")

    plot_curves(raw_metrics, pp_metrics, args.out_png)
    print(f"\nSaved plot: {args.out_png}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Grid search UFold-style postprocessing offset on a validation split.

What it does:
- Loads a trained model checkpoint
- Builds the SAME cluster-based val split as train_with_args.py (if --clstr_path is given)
- Runs inference on val
- For each offset in a grid, computes avg F1/P/R (using UFold postprocess + matching)
- Prints best offset by F1 (and optionally by Precision constraint)

Example:
  python scripts/grid_search_pp_offset.py \
    --data_dir /root/autodl-tmp/newPredicProject/dbnFiles \
    --max_len 600 \
    --clstr_path /root/autodl-tmp/newPredicProject/bprna1m_cdhit90_len600.clstr \
    --train_frac 0.98 --val_frac 0.01 --seed 42 \
    --ckpt /root/autodl-tmp/newPredicProject/checkpoints/model_best.pth \
    --batch_size 2 --device cuda \
    --min_loop 4 --offset_min 0.10 --offset_max 0.80 --offset_step 0.05
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.dataset import MultiFileDatasetUpgrade, collate_pad
from src.model import SpotRNA_LSTM_Refined
from src.metrics import calculate_f1_postprocess_ufold
from scripts.cluster_utils import parse_cd_hit_clusters


def create_cluster_split(dataset, clstr_path, train_frac, val_frac, seed=42):
    random.seed(seed)
    np.random.seed(seed)

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

    train_clusters = cluster_indices[:n_train]
    val_clusters = cluster_indices[n_train:n_train + n_val]

    train_indices = [idx for c in train_clusters for idx in c]
    val_indices = [idx for c in val_clusters for idx in c]
    return train_indices, val_indices


@torch.no_grad()
def eval_offsets(model, loader, device, offsets: List[float], min_loop: int):
    # accum
    f1_sum = {o: 0.0 for o in offsets}
    p_sum = {o: 0.0 for o in offsets}
    r_sum = {o: 0.0 for o in offsets}
    n_batches = 0

    model.eval()
    for seqs, labels, masks in loader:
        n_batches += 1
        seqs = seqs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        logits = model(seqs, mask=masks)

        # compute for each offset
        for o in offsets:
            f1, p, r = calculate_f1_postprocess_ufold(
                logits=logits,
                labels=labels,
                seqs=seqs,
                masks=masks,
                offset=o,
                min_loop=min_loop,
            )
            f1_sum[o] += f1
            p_sum[o] += p
            r_sum[o] += r

    # average per batch (matches your training logging style)
    for o in offsets:
        f1_sum[o] /= max(n_batches, 1)
        p_sum[o] /= max(n_batches, 1)
        r_sum[o] /= max(n_batches, 1)

    return f1_sum, p_sum, r_sum, n_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_len", type=int, default=600)

    # split args (match train_with_args.py)
    ap.add_argument("--clstr_path", type=str, default=None)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    # model args
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--resnet_layers", type=int, default=8)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--lstm_hidden", type=int, default=64)

    # postprocess args
    ap.add_argument("--min_loop", type=int, default=4)
    ap.add_argument("--offset_min", type=float, default=0.10)
    ap.add_argument("--offset_max", type=float, default=0.80)
    ap.add_argument("--offset_step", type=float, default=0.05)

    # optional: choose best with precision constraint
    ap.add_argument("--min_precision", type=float, default=None,
                    help="If set, choose best F1 among offsets with P >= this.")

    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # offsets grid
    offsets = []
    o = args.offset_min
    while o <= args.offset_max + 1e-12:
        offsets.append(round(o, 10))
        o += args.offset_step

    print("Offsets:", offsets)

    # dataset
    ds = MultiFileDatasetUpgrade(args.data_dir, max_len=args.max_len)
    if len(ds) == 0:
        raise RuntimeError("Dataset empty. Check --data_dir")

    if args.clstr_path:
        train_idx, val_idx = create_cluster_split(ds, args.clstr_path, args.train_frac, args.val_frac, args.seed)
        val_ds = Subset(ds, val_idx)
        print(f"Using cluster split: val sequences={len(val_ds)}")
    else:
        # random split fallback
        n = len(ds)
        n_train = int(args.train_frac * n)
        n_val = int(args.val_frac * n)
        g = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds, _ = torch.utils.data.random_split(ds, [n_train, n_val, n - n_train - n_val], generator=g)
        print(f"Using random split: val sequences={len(val_ds)}")

    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad, num_workers=0)

    # model
    cfg = Config()
    cfg.RESNET_LAYERS = args.resnet_layers
    cfg.HIDDEN_DIM = args.hidden_dim
    cfg.LSTM_HIDDEN = args.lstm_hidden
    cfg.DEVICE = device

    model = SpotRNA_LSTM_Refined(cfg).to(device)

    state = torch.load(args.ckpt, map_location=device)
    # support either raw state_dict or {"model_state_dict": ...}
    state = state.get("model_state_dict", state)
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {args.ckpt}")

    f1_map, p_map, r_map, n_batches = eval_offsets(model, loader, device, offsets, args.min_loop)
    print(f"Evaluated {n_batches} batches on val")

    # print table
    print("\nOffset grid results:")
    header = f"{'offset':>8} | {'F1':>7} | {'P':>7} | {'R':>7}"
    print(header)
    print("-" * len(header))
    for o in offsets:
        print(f"{o:8.2f} | {f1_map[o]:7.4f} | {p_map[o]:7.4f} | {r_map[o]:7.4f}")

    # select best
    best_o = None
    best_f1 = -1.0

    for o in offsets:
        if args.min_precision is not None and p_map[o] < args.min_precision:
            continue
        if f1_map[o] > best_f1:
            best_f1 = f1_map[o]
            best_o = o

    if best_o is None:
        print("\nNo offset satisfies min_precision. Try lowering --min_precision or widening grid.")
        return

    print("\nBest offset:")
    print(f"  offset={best_o:.2f}  F1={f1_map[best_o]:.4f}  P={p_map[best_o]:.4f}  R={r_map[best_o]:.4f}")
    if args.min_precision is not None:
        print(f"  (selected with constraint P >= {args.min_precision})")


if __name__ == "__main__":
    main()
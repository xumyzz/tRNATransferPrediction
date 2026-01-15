import argparse
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

# 你需要按你的项目实际路径调整这两个 import
# 目标：复用你训练时的 dataset / dataloader / model 构造
from src.dataset import MultiFileDatasetUpgrade  # 如果你的类名/路径不同请改
from src.model import SpotRNA_LSTM_Refined       # 如果你的类名/路径不同请改
from src.config import Config                    # 如果你不用 Config，也可以删掉


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def masked_f1_from_logits(logits, labels, mask, threshold=0.5):
    """
    logits: (L, L) tensor
    labels: (L, L) tensor {0,1}
    mask:   (L,) tensor {0,1} valid positions
    """
    # sigmoid -> prob
    probs = torch.sigmoid(logits)

    # build 2D mask: valid i and valid j
    m = mask.float()
    m2 = (m[:, None] * m[None, :]).bool()

    # ignore diagonal
    diag = torch.eye(labels.size(0), device=labels.device).bool()
    m2 = m2 & (~diag)

    # (optional) only upper triangle to avoid double counting
    triu = torch.triu(torch.ones_like(labels, dtype=torch.bool), diagonal=1)
    m2 = m2 & triu

    y_true = labels.bool() & m2
    y_pred = (probs >= threshold) & m2

    tp = torch.logical_and(y_pred, y_true).sum().item()
    fp = torch.logical_and(y_pred, ~y_true).sum().item()
    fn = torch.logical_and(~y_pred, y_true).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1, precision, recall, probs


def plot_pairmaps(seq_len, true_map, prob_map, out_png, title):
    # 只画上三角更清楚（可选）
    true_show = true_map.copy()
    prob_show = prob_map.copy()
    for i in range(seq_len):
        true_show[i, :i+1] = np.nan
        prob_show[i, :i+1] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    im0 = axes[0].imshow(true_show, vmin=0, vmax=1, cmap="viridis")
    axes[0].set_title("Ground truth (pairs)")
    axes[0].set_xlabel("j")
    axes[0].set_ylabel("i")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(prob_show, vmin=0, vmax=1, cmap="viridis")
    axes[1].set_title("Predicted P(pair)")
    axes[1].set_xlabel("j")
    axes[1].set_ylabel("i")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory containing .st/.dbn files")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pth)")
    ap.add_argument("--out_dir", default="viz_out", help="Output directory for PNGs")
    ap.add_argument("--max_len", type=int, default=600)
    ap.add_argument("--num_samples", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None, help="cuda/cpu, default auto")
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Build dataset (尽量与训练一致)
    # 如果你的 MultiFileDatasetUpgrade 构造函数参数不同，请按你的实现调整
    ds = MultiFileDatasetUpgrade(
        data_dir=args.data_dir,
        max_len=args.max_len,
        split=None,           # 仅用于展示时通常不需要 split
    )

    if len(ds) == 0:
        raise RuntimeError("Dataset is empty. Check --data_dir / parsing / filters.")

    # 2) Build model
    # 如果你训练时用的不是 Config，请改成你训练脚本里创建 config 的方式
    cfg = Config
    cfg.MAX_LEN = args.max_len
    model = SpotRNA_LSTM_Refined(cfg).to(device)

    # 3) Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    # 兼容两种保存方式：直接 state_dict 或 {"model_state_dict": ...}
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # 4) Sample indices
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:args.num_samples]

    print(f"Visualizing {len(indices)} samples from dataset of size {len(ds)}")
    print(f"Saving outputs to: {args.out_dir}")

    for k, idx in enumerate(indices):
        seqs, labels, masks = ds[idx]  # seqs: (L,4) or (max_len,4), labels:(L,L), masks:(L,)
        # ensure tensors
        if not isinstance(seqs, torch.Tensor):
            seqs = torch.tensor(seqs)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        if not isinstance(masks, torch.Tensor):
            masks = torch.tensor(masks)

        # add batch dim
        seqs_b = seqs.unsqueeze(0).to(device)
        labels_b = labels.to(device)
        masks_b = masks.unsqueeze(0).to(device)

        with torch.no_grad():
            logits_b = model(seqs_b, mask=masks_b)  # (1,L,L)

        logits = logits_b[0]
        L = int(masks.sum().item())

        f1, p, r, probs = masked_f1_from_logits(logits[:L, :L], labels_b[:L, :L], masks[:L].to(device), threshold=args.threshold)

        true_map = to_numpy(labels_b[:L, :L].float())
        prob_map = to_numpy(probs[:L, :L].float())

        out_png = os.path.join(args.out_dir, f"sample_{k:03d}_idx{idx}_F1_{f1:.3f}.png")
        title = f"idx={idx}  L={L}  F1={f1:.3f}  P={p:.3f}  R={r:.3f}  thr={args.threshold}"
        plot_pairmaps(L, true_map, prob_map, out_png, title)

        print(title, "->", out_png)


if __name__ == "__main__":
    main()
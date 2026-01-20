0import argparse
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

# 你需要按你的项目实际路径调整 import
from src.dataset import MultiFileDatasetUpgrade  # 如果不同请改
from src.model import SpotRNA_LSTM_Refined       # 如果不同请改
from src.config import Config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_numpy(x):
    return x.detach().cpu().numpy()


def infer_mask_from_onehot(seqs_1d: torch.Tensor) -> torch.Tensor:
    """
    seqs_1d: (L, 4) one-hot/padded
    return:  (L,) bool mask, True means valid position
    规则：如果这一行全是 0，当成 PAD
    """
    return (seqs_1d.abs().sum(dim=-1) > 0)


def masked_counts_from_logits(logits, labels, mask, threshold=0.5):
    """
    logits: (L, L)
    labels: (L, L) 0/1
    mask:   (L,) bool
    """
    probs = torch.sigmoid(logits)

    m = mask.bool()
    m2 = (m[:, None] & m[None, :])

    # ignore diagonal
    diag = torch.eye(labels.size(0), device=labels.device).bool()
    m2 = m2 & (~diag)

    # only upper triangle to avoid double counting
    triu = torch.triu(torch.ones_like(labels, dtype=torch.bool), diagonal=1)
    m2 = m2 & triu

    y_true = (labels > 0.5) & m2
    y_pred = (probs >= threshold) & m2

    tp = torch.logical_and(y_pred, y_true).sum().item()
    fp = torch.logical_and(y_pred, ~y_true).sum().item()
    fn = torch.logical_and(~y_pred, y_true).sum().item()
    return tp, fp, fn, probs


def prf_from_counts(tp, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def plot_pairmaps(L, true_map, prob_map, out_png, title):
    # show upper triangle only (optional)
    true_show = true_map.copy()
    prob_show = prob_map.copy()
    for i in range(L):
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


def unpack_dataset_item(item):
    """
    兼容:
      (seqs, labels)
      (seqs, labels, masks)
      dict 形式（有的项目会这样）
    """
    if isinstance(item, dict):
        seqs = item["seqs"]
        labels = item["labels"]
        masks = item.get("masks", None)
        return seqs, labels, masks

    if isinstance(item, (tuple, list)):
        if len(item) == 2:
            return item[0], item[1], None
        if len(item) == 3:
            return item[0], item[1], item[2]

    raise ValueError(f"Unsupported dataset item type/len: {type(item)} / {getattr(item, '__len__', lambda: 'NA')()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="viz_out")
    ap.add_argument("--max_len", type=int, default=600)
    ap.add_argument("--num_samples", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = MultiFileDatasetUpgrade(data_dir_or_file=args.data_dir, max_len=args.max_len)

    cfg = Config
    cfg.MAX_LEN = args.max_len
    model = SpotRNA_LSTM_Refined(cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:args.num_samples]

    print(f"Visualizing {len(indices)} samples from dataset size {len(ds)}")
    print(f"Saving to {args.out_dir}")

    for k, idx in enumerate(indices):
        item = ds[idx]
        seqs, labels, masks = unpack_dataset_item(item)

        if not isinstance(seqs, torch.Tensor):
            seqs = torch.tensor(seqs)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        if masks is not None and not isinstance(masks, torch.Tensor):
            masks = torch.tensor(masks)

        # 你模型要求输入 (B,L,4)
        if seqs.dim() == 2:
            seqs_b = seqs.unsqueeze(0)
        else:
            raise ValueError(f"Expected seqs dim=2 (L,4). Got {seqs.shape}")

        # mask：如果 dataset 没给，就从 one-hot 推断
        if masks is None:
            mask_1d = infer_mask_from_onehot(seqs)  # (L,)
        else:
            mask_1d = masks.bool() if masks.dim() == 1 else masks[0].bool()

        L = int(mask_1d.sum().item())
        if L <= 1:
            print(f"Skip idx={idx} because valid length L={L}")
            continue

        seqs_b = seqs_b.to(device)
        labels = labels.to(device)
        mask_1d = mask_1d.to(device)

        with torch.no_grad():
            logits = model(seqs_b)[0]  # (L,L) padded

        # crop to valid length for clean plotting & metrics
        logits_v = logits[:L, :L]
        labels_v = labels[:L, :L]
        mask_v = mask_1d[:L]

        tp, fp, fn, probs = masked_counts_from_logits(logits_v, labels_v, mask_v, threshold=args.threshold)
        precision, recall, f1 = prf_from_counts(tp, fp, fn)

        true_map = to_numpy(labels_v.float())
        prob_map = to_numpy(probs.float())

        out_png = os.path.join(args.out_dir, f"sample_{k:03d}_idx{idx}_F1_{f1:.3f}.png")
        title = f"idx={idx} L={L} F1={f1:.3f} P={precision:.3f} R={recall:.3f} thr={args.threshold}"
        plot_pairmaps(L, true_map, prob_map, out_png, title)
        print(title, "->", out_png)


if __name__ == "__main__":
    main()
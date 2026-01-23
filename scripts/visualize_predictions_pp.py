import argparse
import os
import random
import inspect

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.dataset import MultiFileDatasetUpgrade
from src.model import SpotRNA_LSTM_Refined
from src.config import Config

from src.metrics import calculate_f1_postprocess_ufold
from src.postprocess_ufold import (
    build_M,
    transform_Y,
    max_weight_matching_decode,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_numpy(x):
    return x.detach().cpu().numpy()


def infer_mask_from_onehot(seqs_1d: torch.Tensor) -> torch.Tensor:
    return (seqs_1d.abs().sum(dim=-1) > 0)


def onehot_to_seq_str(seqs_1d: torch.Tensor) -> str:
    """
    Assumes one-hot channel order [A, C, G, U].
    If your order differs, change mapping.
    """
    if not isinstance(seqs_1d, torch.Tensor):
        seqs_1d = torch.tensor(seqs_1d)

    if seqs_1d.dim() != 2 or seqs_1d.size(-1) != 4:
        raise ValueError(f"Expected (L,4) one-hot. Got {tuple(seqs_1d.shape)}")

    row_sum = seqs_1d.abs().sum(dim=-1)
    idx = torch.argmax(seqs_1d, dim=-1)

    mapping = np.array(list("ACGU"))
    chars = []
    for i in range(seqs_1d.size(0)):
        if row_sum[i].item() == 0:
            chars.append("N")
        else:
            chars.append(mapping[int(idx[i].item())])
    return "".join(chars)


def masked_counts_from_logits(logits, labels, mask, threshold=0.5):
    probs = torch.sigmoid(logits)

    m = mask.bool()
    m2 = (m[:, None] & m[None, :])

    diag = torch.eye(labels.size(0), device=labels.device).bool()
    m2 = m2 & (~diag)

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


def unpack_dataset_item(item):
    if isinstance(item, dict):
        return item["seqs"], item["labels"], item.get("masks", None)

    if isinstance(item, (tuple, list)):
        if len(item) == 2:
            return item[0], item[1], None
        if len(item) == 3:
            return item[0], item[1], item[2]

    raise ValueError(f"Unsupported dataset item type/len: {type(item)}")


def plot_pairmaps_3col(L, true_map, prob_map, post_map, out_png, title):
    def upper_nan(x):
        y = x.copy()
        for i in range(L):
            y[i, : i + 1] = np.nan
        return y

    true_show = upper_nan(true_map)
    prob_show = upper_nan(prob_map)
    post_show = upper_nan(post_map)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

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

    im2 = axes[2].imshow(post_show, vmin=0, vmax=1, cmap="viridis")
    axes[2].set_title("Postprocess A (matching)")
    axes[2].set_xlabel("j")
    axes[2].set_ylabel("i")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def build_M_compat(seq_str: str, min_loop: int, device: torch.device) -> torch.Tensor:
    """
    build_M in your project may:
    - accept (seq_str) only OR (seq_str, min_loop=...)
    - return numpy array / list / torch tensor
    This wrapper normalizes it into torch.Tensor on device.
    """
    sig = inspect.signature(build_M)
    if "min_loop" in sig.parameters:
        M = build_M(seq_str, min_loop=min_loop)
    else:
        M = build_M(seq_str)

    if isinstance(M, torch.Tensor):
        return M.to(device)
    return torch.tensor(M, device=device)


@torch.no_grad()
def decode_postprocess_A(
    logits_v: torch.Tensor,
    seq_onehot_v: torch.Tensor,
    offset: float,
    min_loop: int,
) -> torch.Tensor:
    device = logits_v.device
    Y = torch.sigmoid(logits_v)

    seq_str = onehot_to_seq_str(seq_onehot_v.detach().cpu())
    M = build_M_compat(seq_str, min_loop=min_loop, device=device)

    S = transform_Y(Y, M)
    if not isinstance(S, torch.Tensor):
        S = torch.tensor(S, device=device)

    A = max_weight_matching_decode(S, offset=offset)
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, device=device)
    A = A.to(device)

    A = ((A + A.t()) > 0).to(torch.float32)
    A.fill_diagonal_(0)
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="viz_out")
    ap.add_argument("--max_len", type=int, default=600)
    ap.add_argument("--num_samples", type=int, default=12)

    ap.add_argument("--threshold", type=float, default=0.5)

    ap.add_argument("--pp_offset", type=float, default=0.35)
    ap.add_argument("--min_loop", type=int, default=4)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ds = MultiFileDatasetUpgrade(data_dir_or_file=args.data_dir, max_len=args.max_len)

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

    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[: args.num_samples]

    print(f"Visualizing {len(indices)} samples from dataset size {len(ds)}")
    print(f"Saving to {args.out_dir}")

    for k, idx in enumerate(indices):
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
            print(f"Skip idx={idx} because valid length L={L}")
            continue

        seqs_b = seqs.unsqueeze(0).to(device)
        labels = labels.to(device)
        mask_1d = mask_1d.to(device)

        with torch.no_grad():
            logits = model(seqs_b)[0]

        logits_v = logits[:L, :L]
        labels_v = labels[:L, :L]
        seqs_v = seqs[:L, :].to(device)
        mask_v = mask_1d[:L]

        tp, fp, fn, probs = masked_counts_from_logits(
            logits_v, labels_v, mask_v, threshold=args.threshold
        )
        raw_p, raw_r, raw_f1 = prf_from_counts(tp, fp, fn)

        pp_f1, pp_p, pp_r = calculate_f1_postprocess_ufold(
            logits=logits_v.unsqueeze(0),
            labels=labels_v.unsqueeze(0),
            seqs=seqs_v.unsqueeze(0),
            masks=mask_v.unsqueeze(0),
            offset=args.pp_offset,
            min_loop=args.min_loop,
        )

        post_A = decode_postprocess_A(
            logits_v=logits_v,
            seq_onehot_v=seqs_v,
            offset=args.pp_offset,
            min_loop=args.min_loop,
        )

        true_map = to_numpy(labels_v.float())
        prob_map = to_numpy(probs.float())
        post_map = to_numpy(post_A.float())

        out_png = os.path.join(
            args.out_dir,
            f"sample_{k:03d}_idx{idx}_L{L}_rawF1_{raw_f1:.3f}_ppF1_{pp_f1:.3f}.png",
        )
        title = (
            f"idx={idx} L={L} | "
            f"RAW thr={args.threshold}: F1={raw_f1:.3f} P={raw_p:.3f} R={raw_r:.3f} | "
            f"PP offset={args.pp_offset} min_loop={args.min_loop}: F1={pp_f1:.3f} P={pp_p:.3f} R={pp_r:.3f}"
        )

        plot_pairmaps_3col(L, true_map, prob_map, post_map, out_png, title)
        print(title, "->", out_png)


if __name__ == "__main__":
    main()
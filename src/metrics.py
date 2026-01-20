import numpy as np
import torch

from src.postprocess_ufold import (
    onehot_to_seq,
    build_M,
    transform_Y,
    max_weight_matching_decode,
)


def f1_from_binary_contact(A: np.ndarray, labels: np.ndarray):
    """
    A: (L,L) 0/1 predicted, symmetric
    labels: (L,L) 0/1 true
    只算上三角避免双计
    """
    triu = np.triu(np.ones_like(A, dtype=bool), k=1)
    y_pred = (A > 0) & triu
    y_true = (labels > 0) & triu

    tp = np.logical_and(y_pred, y_true).sum()
    fp = np.logical_and(y_pred, ~y_true).sum()
    fn = np.logical_and(~y_pred, y_true).sum()

    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return f1, p, r, tp, fp, fn


def calculate_f1_postprocess_ufold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    seqs: torch.Tensor,
    masks: torch.Tensor = None,
    offset: float = 0.5,
    min_loop: int = 4,
):
    """
    UFold-style postprocessing metric.

    Args:
        logits: (B,L,L) model outputs (logits)
        labels: (B,L,L) ground truth {0,1}
        seqs:   (B,L,4) one-hot
        masks:  (B,L) 0/1 valid positions (optional)
        offset: threshold on constrained probabilities after T(Y)
        min_loop: disallow |i-j| < min_loop

    Returns:
        (avg_f1, avg_p, avg_r) averaged over batch
    """
    B, L, _ = logits.shape

    logits = logits.detach().cpu()
    labels = labels.detach().cpu()
    seqs = seqs.detach().cpu()
    if masks is not None:
        masks = masks.detach().cpu()

    f1_sum = 0.0
    p_sum = 0.0
    r_sum = 0.0
    used = 0

    for b in range(B):
        if masks is None:
            valid = (seqs[b].abs().sum(dim=-1) > 0)
        else:
            valid = masks[b].bool()

        Lv = int(valid.sum().item())
        if Lv <= 1:
            continue

        # probs
        Y = torch.sigmoid(logits[b, :Lv, :Lv]).numpy()
        y_true = labels[b, :Lv, :Lv].numpy()

        # sequence string for constraints
        seq_str = onehot_to_seq(seqs[b, :Lv].numpy())

        # constraints
        M = build_M(seq_str, min_loop=min_loop)
        S = transform_Y(Y, M)

        # non-overlapping decode (matching)
        A = max_weight_matching_decode(S, offset=offset)

        f1, p, r, *_ = f1_from_binary_contact(A, y_true)
        f1_sum += f1
        p_sum += p
        r_sum += r
        used += 1

    if used == 0:
        return 0.0, 0.0, 0.0

    return f1_sum / used, p_sum / used, r_sum / used
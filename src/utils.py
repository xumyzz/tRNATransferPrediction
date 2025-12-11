import torch


def compute_masked_loss(logits, targets, masks, criterion):
    """
    计算带 Mask 的 Loss
    :param criterion: 也就是你的 criterion_raw (BCEWithLogitsLoss)
    """
    # 构建 2D Mask: (B, L, L)
    mask_2d = masks.unsqueeze(1) * masks.unsqueeze(2)

    # 确保设备一致
    device = logits.device
    mask_2d = mask_2d.to(device)
    targets = targets.to(device)

    # 计算 Element-wise Loss
    loss_mat = criterion(logits, targets)

    # 只保留 Mask 内的 Loss 并求平均
    # 加上 1e-6 防止除以 0
    loss = (loss_mat * mask_2d).sum() / (mask_2d.sum() + 1e-6)
    return loss


def calculate_f1(logits, labels, masks):
    """
    计算 F1 分数
    """
    mask_2d = masks.unsqueeze(1) * masks.unsqueeze(2)

    # 预测概率转二值
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float() * mask_2d

    # 计算 TP, FP, FN
    tp = (preds * labels).sum()
    fp = (preds * (1 - labels)).sum()
    fn = ((1 - preds) * labels).sum()

    # 计算 F1 (加 1e-8 防止除以 0)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return f1.item()
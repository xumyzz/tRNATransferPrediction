import torch
from torch import nn


def compute_masked_loss(logits, targets, masks, pos_weight=None):
    """
    计算带 Mask 的 Loss，并在内部处理类别不平衡权重
    Args:
        logits: 模型输出 (B, L, L)
        targets: 真实标签 (B, L, L)
        masks: 序列 Mask (B, L)
        pos_weight: 正样本权重数值 (float). 如果为 None, 则默认设为 1.0
    """
    # 1. 处理默认权重
    if pos_weight is None:
        pos_weight = 1.0

    # 2. 构建 2D Mask: (B, L) -> (B, L, L)
    # 只有当 i 和 j 都在 mask 内时，(i, j) 才是有效的
    mask_2d = masks.unsqueeze(1) * masks.unsqueeze(2)

    # 3. 确保设备一致
    device = logits.device
    mask_2d = mask_2d.to(device)
    targets = targets.to(device)

    # 4. 动态定义 Loss 函数
    # reduction='none' 是为了得到每个像素的 loss，以便后续乘以 mask
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device),
        reduction='none'
    )

    # 5. 计算 Loss
    loss_mat = criterion(logits, targets)

    # 6. 只保留 Mask 内的 Loss 并求平均
    # (mask_2d.sum() + 1e-6) 防止除以 0
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
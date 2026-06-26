#!/usr/bin/env python3
"""
SpotRNA-V2 标准训练 — 使用 bpRNA 官方 TR0/VL0/TS0 划分
=========================================================
与 SPOT-RNA / UFold / MXfold2 / KnotFold 完全对齐的数据划分。

用法:
    # 从零训练 V2 模型 (单张 4090, ~50 epochs)
    python experiments/train_v2_standard.py \
        --train_dir data/TR0 \
        --val_dir data/VL0 \
        --save_dir checkpoints_v2_tr0 \
        --epochs 50 --device cuda:0

    # 从已有 checkpoint 继续训练
    python experiments/train_v2_standard.py \
        --train_dir data/TR0 --val_dir data/VL0 \
        --pretrained checkpoints_v2_tr0/model_best.pth \
        --save_dir checkpoints_v2_tr0_resume \
        --epochs 20 --lr 5e-5 --device cuda:0

    # 评测 TS0
    python experiments/ts0_evaluate.py \
        --checkpoint checkpoints_v2_tr0/model_best.pth \
        --ts0_dir data/TS0 --output_dir results/ts0_v2_tr0
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.dataset import MultiFileDatasetUpgrade, collate_pad
from src.model import SpotRNA_LSTM_Refined_BPPM_Chimeric


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   ImprovedRNA Loss (同 train_improved.py)               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ImprovedRNALoss(nn.Module):
    def __init__(
        self,
        base_pos_weight: float = 2.0,
        w_dice: float = 0.1,
        w_margin: float = 0.05,
        margin_threshold: float = 0.6,
        label_smoothing: float = 0.05,
        distance_aware: bool = True,
        max_distance: int = 600,
    ):
        super().__init__()
        self.base_pos_weight = base_pos_weight
        self.w_dice = w_dice
        self.w_margin = w_margin
        self.margin_threshold = margin_threshold
        self.label_smoothing = label_smoothing
        self.distance_aware = distance_aware
        self.max_distance = max_distance

    def forward(self, logits, targets, valid_mask):
        B, L, _ = logits.shape
        device = logits.device

        targets_smooth = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        targets_smooth = targets_smooth * valid_mask
        targets_binary = targets * valid_mask

        if self.distance_aware and L > 10:
            pos_i = torch.arange(L, device=device).unsqueeze(1)
            pos_j = torch.arange(L, device=device).unsqueeze(0)
            distance = (pos_i - pos_j).abs().float()
            dist_log = torch.log1p(distance / 10.0)
            pos_weight_matrix = self.base_pos_weight * (1.0 + dist_log)
        else:
            pos_weight_matrix = self.base_pos_weight * torch.ones(L, L, device=device)

        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        weight_map = torch.where(targets_binary > 0.5,
                                 pos_weight_matrix.unsqueeze(0),
                                 torch.ones_like(bce))
        bce_weighted = (bce * weight_map * valid_mask).sum()
        denom = (weight_map * valid_mask).sum().clamp(min=1.0)
        bce_loss = bce_weighted / denom

        dice_loss = torch.tensor(0.0, device=device)
        if self.w_dice > 0:
            pred_prob = torch.sigmoid(logits) * valid_mask
            inter = (pred_prob * targets_binary).sum(dim=(-2, -1))
            pred_sum = pred_prob.sum(dim=(-2, -1))
            target_sum = targets_binary.sum(dim=(-2, -1))
            dice_loss = (1.0 - (2.0 * inter + 1e-6) / (pred_sum + target_sum + 1e-6)).mean()

        margin_loss = torch.tensor(0.0, device=device)
        if self.w_margin > 0:
            pred_prob = torch.sigmoid(logits)
            pos_mask = (targets_binary > 0.5).float()
            margin = F.relu(self.margin_threshold - pred_prob) * pos_mask * valid_mask
            margin_loss = margin.sum() / (pos_mask * valid_mask).sum().clamp(min=1.0)

        total = bce_loss + self.w_dice * dice_loss + self.w_margin * margin_loss
        return total, {
            'bce': bce_loss.item(),
            'dice': dice_loss.item() if isinstance(dice_loss, torch.Tensor) else dice_loss,
            'margin': margin_loss.item() if isinstance(margin_loss, torch.Tensor) else margin_loss,
            'total': total.item(),
        }


def build_valid_mask(masks, L, device):
    """构建 2D valid_mask (排除 padding、对角线)"""
    B = masks.shape[0]

    if masks.dim() == 4:
        mask_1d = masks[:, 0, 0, :]
    elif masks.dim() == 3:
        mask_1d = masks.squeeze(-1)
    elif masks.dim() == 2:
        mask_1d = masks
    else:
        mask_1d = masks.view(B, -1)
    if mask_1d.dim() == 1:
        mask_1d = mask_1d.unsqueeze(0).expand(B, -1)

    valid = mask_1d.unsqueeze(2) * mask_1d.unsqueeze(1)
    diag = torch.eye(L, device=device).unsqueeze(0)
    valid = valid * (1.0 - diag)
    return valid.float()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          训练主函数                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train(args):
    print("=" * 70)
    print("  SpotRNA-V2 标准训练 — TR0 → VL0 → TS0")
    print("=" * 70)
    print(f"  Train: {args.train_dir}")
    print(f"  Val:   {args.val_dir}")
    print(f"  Save:  {args.save_dir}")
    print(f"  Device: {args.device}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ═══════════════════════════════════════════════════════════════
    # 1. 加载数据
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[1/4] 加载数据集...")
    train_ds = MultiFileDatasetUpgrade(args.train_dir, max_len=args.max_len)
    val_ds = MultiFileDatasetUpgrade(args.val_dir, max_len=args.max_len)
    print(f"  Train: {len(train_ds)} 条序列")
    print(f"  Val:   {len(val_ds)} 条序列")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_pad, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_pad, num_workers=args.num_workers)

    # ═══════════════════════════════════════════════════════════════
    # 2. 模型 / 损失 / 优化器
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[2/4] 初始化模型...")
    config = Config()
    config.RESNET_LAYERS = args.resnet_layers
    config.HIDDEN_DIM = args.hidden_dim
    config.LSTM_HIDDEN = args.lstm_hidden

    model = SpotRNA_LSTM_Refined_BPPM_Chimeric(config)

    if args.pretrained and os.path.exists(args.pretrained):
        print(f"  加载预训练权重: {args.pretrained}")
        state_dict = torch.load(args.pretrained, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        print(f"  加载完成")

    model = model.to(args.device)

    criterion = ImprovedRNALoss(
        base_pos_weight=args.pos_weight,
        w_dice=args.w_dice,
        w_margin=args.w_margin,
        margin_threshold=args.margin_threshold,
        label_smoothing=args.label_smoothing,
        distance_aware=args.distance_aware,
        max_distance=args.max_len,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    eff_bs = args.batch_size * args.accum_steps
    print(f"  模型: SpotRNA_LSTM_Refined_BPPM_Chimeric")
    print(f"  Loss: ImprovedRNALoss (pos_w={args.pos_weight}, dice={args.w_dice}, "
          f"margin={args.w_margin}, smooth={args.label_smoothing})")
    print(f"  Optimizer: AdamW lr={args.lr}, wd={args.weight_decay}")
    print(f"  Effective Batch: {eff_bs} (bs={args.batch_size} × accum={args.accum_steps})")

    # ═══════════════════════════════════════════════════════════════
    # 3. 训练循环
    # ═══════════════════════════════════════════════════════════════
    os.makedirs(args.save_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)

    best_val_f1 = 0.0
    best_epoch = 0
    nan_count = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = train_bce = train_dice = train_margin = 0.0
        valid_batches = 0

        for step, (seqs, bppms, labels, masks) in enumerate(train_loader):
            seqs = seqs.to(args.device)
            bppms = bppms.to(args.device)
            labels = labels.to(args.device)
            masks = masks.to(args.device)
            B, L, _ = seqs.shape

            # 前向 (bpRNA 模式: 域标签全 0)
            logits = model(seqs, bppm=bppms, mask=masks,
                           trna_5end_len=None, trna_3end_len=None)

            valid_mask = build_valid_mask(masks, L, args.device)

            loss, comps = criterion(logits, labels, valid_mask)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue

            loss = loss / args.accum_steps
            loss.backward()

            train_loss += comps['total']
            train_bce += comps['bce']
            train_dice += comps['dice']
            train_margin += comps['margin']
            valid_batches += 1

            if (step + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 100 == 0:
                print(f"  Epoch [{epoch+1}/{args.epochs}] Step [{step:5d}] "
                      f"Loss={comps['total']:.4f} "
                      f"(BCE={comps['bce']:.4f} D={comps['dice']:.4f} M={comps['margin']:.4f})",
                      flush=True)

        # 清理残余梯度
        if valid_batches % args.accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        # ── Epoch 汇总 ──
        n = max(valid_batches, 1)
        print(f"\n  ── Epoch {epoch+1} Train ──")
        print(f"  Loss={train_loss/n:.4f} BCE={train_bce/n:.4f} "
              f"Dice={train_dice/n:.4f} Margin={train_margin/n:.4f} "
              f"LR={lr_now:.2e} NaN={nan_count}")

        # ═══════════════════════════════════════════════════════════
        # 4. Validation
        # ═══════════════════════════════════════════════════════════
        model.eval()
        val_loss = val_bce = val_dice = 0.0
        val_f1_sum = val_p_sum = val_r_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for seqs, bppms, labels, masks in val_loader:
                seqs = seqs.to(args.device)
                bppms = bppms.to(args.device)
                labels = labels.to(args.device)
                masks = masks.to(args.device)
                Bv, Lv = seqs.shape[0], seqs.shape[1]

                logits = model(seqs, bppm=bppms, mask=masks,
                               trna_5end_len=None, trna_3end_len=None)

                valid_mask_v = build_valid_mask(masks, Lv, args.device)
                loss, comps = criterion(logits, labels, valid_mask_v)
                val_loss += comps['total']
                val_bce += comps['bce']
                val_dice += comps['dice']
                val_n += 1

                # F1
                probs = torch.sigmoid(logits)
                if masks.dim() == 4:
                    m1d = masks[:, 0, 0, :]
                elif masks.dim() == 2:
                    m1d = masks
                else:
                    m1d = masks.squeeze()
                m2d = m1d.unsqueeze(2) * m1d.unsqueeze(1)
                tril = torch.tril(torch.ones(Lv, Lv, device=args.device),
                                  diagonal=-1).bool().unsqueeze(0)
                pred_bin = (probs > args.pp_offset).float() * m2d * tril
                lbl_eval = labels * m2d * tril

                TP = (pred_bin * lbl_eval).sum().item()
                FP = (pred_bin * (1 - lbl_eval)).sum().item()
                FN = ((1 - pred_bin) * lbl_eval).sum().item()
                p = TP / (TP + FP + 1e-8)
                r = TP / (TP + FN + 1e-8)
                f1 = 2 * p * r / (p + r + 1e-8)
                val_f1_sum += f1
                val_p_sum += p
                val_r_sum += r

        vn = max(val_n, 1)
        avg_val_f1 = val_f1_sum / vn
        avg_val_p = val_p_sum / vn
        avg_val_r = val_r_sum / vn

        print(f"  ── Epoch {epoch+1} Val ──")
        print(f"  Loss={val_loss/vn:.4f} BCE={val_bce/vn:.4f} Dice={val_dice/vn:.4f}")
        print(f"  F1={avg_val_f1:.4f}  P={avg_val_p:.4f}  R={avg_val_r:.4f}")

        # ── 保存 ──
        ckpt_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1:03d}.pth")
        torch.save(model.state_dict(), ckpt_path)

        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            best_epoch = epoch + 1
            best_path = os.path.join(args.save_dir, "model_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  ★ 新最佳! F1={best_val_f1:.4f} P={avg_val_p:.4f} R={avg_val_r:.4f} "
                  f"(epoch {best_epoch})")

    print(f"\n{'=' * 70}")
    print(f"  训练完成!")
    print(f"  Best Val F1: {best_val_f1:.4f} (epoch {best_epoch}, VL0)")
    print(f"  模型: {os.path.join(args.save_dir, 'model_best.pth')}")
    print(f"  评测 TS0: python experiments/ts0_evaluate.py \\")
    print(f"    --checkpoint {args.save_dir}/model_best.pth \\")
    print(f"    --ts0_dir data/TS0 --output_dir results/ts0_v2_tr0")
    print(f"{'=' * 70}")

    return best_val_f1


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          命令行入口                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description='SpotRNA-V2 标准训练 (TR0→VL0→TS0)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 数据
    parser.add_argument('--train_dir', type=str, default='data/TR0')
    parser.add_argument('--val_dir', type=str, default='data/VL0')
    parser.add_argument('--max_len', type=int, default=600)

    # 训练
    parser.add_argument('--save_dir', type=str, default='checkpoints_v2_tr0')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--accum_steps', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)

    # 模型
    parser.add_argument('--resnet_layers', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lstm_hidden', type=int, default=64)

    # 损失
    parser.add_argument('--pos_weight', type=float, default=2.0)
    parser.add_argument('--w_dice', type=float, default=0.1)
    parser.add_argument('--w_margin', type=float, default=0.05)
    parser.add_argument('--margin_threshold', type=float, default=0.6)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--distance_aware', type=int, default=1)

    # 评估
    parser.add_argument('--pp_offset', type=float, default=0.5)

    # 设备
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train(args)


if __name__ == '__main__':
    main()

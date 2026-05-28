#!/usr/bin/env python3
"""
Enhanced training script with Chimeric RNA domain isolation support.
"""

import os
import sys
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.dataset import MultiFileDatasetUpgrade, collate_pad
from src.model import (
    SpotRNA_LSTM_Refined,
    SpotRNA_LSTM_Refined_Attention,
    SpotRNA_LSTM_Refined_BPPM,
    SpotRNA_LSTM_Refined_BPPM_Chimeric
)
from scripts.cluster_utils import parse_cd_hit_clusters
from src.metrics import calculate_f1_postprocess_ufold


class ChimericRNALoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, w_dice=0.3,
                 lambda_cross=5.0, lambda_diag=1.0, pos_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.w_dice = w_dice
        self.lambda_cross = lambda_cross
        self.lambda_diag = lambda_diag
        self.pos_weight = pos_weight

    def forward(self, logits, targets, masks, seg_ids=None, debug=False):
        B, L, _ = logits.shape
        device = logits.device

        if masks.dim() == 4:
            mask_1d = masks[:, 0, 0, :]
        elif masks.dim() == 3:
            mask_1d = masks.squeeze(-1) if masks.shape[-1] == 1 else masks[:, :, 0]
        elif masks.dim() == 2:
            mask_1d = masks
        else:
            mask_1d = masks.view(B, -1) if masks.numel() % B == 0 else masks.squeeze()
            if mask_1d.dim() == 1:
                mask_1d = mask_1d.unsqueeze(0).expand(B, -1)

        mask_2d = mask_1d.unsqueeze(2) * mask_1d.unsqueeze(1)
        if mask_2d.shape != (B, L, L):
            mask_2d = mask_2d.view(B, L, L)

        valid_mask = mask_2d.float().clone()
        diag = torch.eye(L, device=device).unsqueeze(0)
        valid_mask = valid_mask * (1.0 - diag.float())

        if seg_ids is not None:
            seg_i = seg_ids.unsqueeze(2)
            seg_j = seg_ids.unsqueeze(1)
            same_domain = (seg_i == seg_j).float()
            valid_mask = valid_mask * same_domain

        valid_sum = valid_mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)

        targets_safe = targets * valid_mask
        logits_safe = torch.clamp(logits, min=-15.0, max=15.0)

        pw = torch.tensor([self.pos_weight], device=device)
        bce = F.binary_cross_entropy_with_logits(
            logits_safe, targets_safe, reduction='none', pos_weight=pw
        )
        bce = bce * valid_mask
        bce_stable = bce.clamp(max=50.0)
        pt = torch.exp(-bce_stable)
        focal = self.alpha * (1 - pt) ** self.gamma * bce_stable
        focal_loss = focal.sum() / valid_sum.sum()

        pred_prob = torch.sigmoid(logits_safe) * valid_mask
        targets_masked = targets_safe

        intersection = (pred_prob * targets_masked).sum(dim=(-2, -1))
        pred_sum = pred_prob.sum(dim=(-2, -1))
        target_sum = targets_masked.sum(dim=(-2, -1))
        dice = 1.0 - (2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
        dice_loss = dice.mean()

        loss = focal_loss + self.w_dice * dice_loss

        diag_penalty = (pred_prob * diag).sum(dim=(-2, -1))
        loss = loss + self.lambda_diag * diag_penalty.mean()

        if seg_ids is not None:
            cross_domain = (seg_i != seg_j).float()
            cross_penalty = (pred_prob * cross_domain).sum(dim=(-2, -1))
            loss = loss + self.lambda_cross * cross_penalty.mean()

        if debug:
            cp = cross_penalty.mean().item() if seg_ids is not None else 0
            print(f"  [LossDebug] focal={focal_loss.item():.2f}, dice={dice_loss.item():.4f}, "
                  f"diag_pen={diag_penalty.mean().item():.4f}, cross_pen={cp:.4f}, "
                  f"valid_ratio={valid_mask.mean().item():.3f}")

        return loss


def get_seg_ids(B, L, domain_split_idx, device):
    # 【修复】domain_split_idx=0 时直接返回全零（无隔离）
    if domain_split_idx is None or domain_split_idx == 0:
        return torch.zeros(B, L, dtype=torch.long, device=device)
    
    if isinstance(domain_split_idx, int):
        domain_split_idx = torch.full((B,), domain_split_idx, dtype=torch.long, device=device)
    elif domain_split_idx.dim() == 0:
        domain_split_idx = domain_split_idx.unsqueeze(0).expand(B)

    seg_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    for b in range(B):
        split = domain_split_idx[b].item()
        split = max(1, min(split, L - 1))
        seg_ids[b, split:] = 2
    return seg_ids


def create_cluster_split(dataset, clstr_path, train_frac, val_frac, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    clusters = parse_cd_hit_clusters(clstr_path)
    print(f"Found {len(clusters)} clusters")
    name_to_idx = {name: idx for idx, name in enumerate(dataset.names)}
    cluster_indices = []
    missing_count = 0
    for cluster in clusters:
        indices = []
        for name in cluster:
            if name in name_to_idx:
                indices.append(name_to_idx[name])
            else:
                missing_count += 1
        if indices:
            cluster_indices.append(indices)
    if missing_count > 0:
        print(f"Warning: {missing_count} sequences in clusters not found in dataset")
    print(f"Mapped to {len(cluster_indices)} non-empty clusters")
    random.shuffle(cluster_indices)
    n_clusters = len(cluster_indices)
    n_train = int(n_clusters * train_frac)
    n_val = int(n_clusters * val_frac)
    train_clusters = cluster_indices[:n_train]
    val_clusters = cluster_indices[n_train:n_train + n_val]
    test_clusters = cluster_indices[n_train + n_val:]
    train_indices = [idx for cluster in train_clusters for idx in cluster]
    val_indices = [idx for cluster in val_clusters for idx in cluster]
    test_indices = [idx for cluster in test_clusters for idx in cluster]
    split_info = {
        'total_sequences': len(dataset),
        'total_clusters': n_clusters,
        'train_clusters': len(train_clusters),
        'val_clusters': len(val_clusters),
        'test_clusters': len(test_clusters),
        'train_sequences': len(train_indices),
        'val_sequences': len(val_indices),
        'test_sequences': len(test_indices),
        'train_frac_actual': len(train_indices) / len(dataset),
        'val_frac_actual': len(val_indices) / len(dataset),
        'test_frac_actual': len(test_indices) / len(dataset),
    }
    return (train_indices, val_indices, test_indices), split_info


def train_model(args):
    print(f"Using device: {args.device}")
    print(f"Configuration: {vars(args)}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\nLoading dataset from {args.data_dir}...")
    full_ds = MultiFileDatasetUpgrade(args.data_dir, max_len=args.max_len)
    if len(full_ds) == 0:
        print("Error: No data loaded. Check path.")
        return
    print(f"Loaded {len(full_ds)} sequences")

    if args.clstr_path:
        print(f"\nCreating cluster-based split from {args.clstr_path}...")
        (train_idx, val_idx, test_idx), split_info = create_cluster_split(
            full_ds, args.clstr_path, args.train_frac, args.val_frac, args.seed
        )
        print("\nSplit information:")
        for key, value in split_info.items():
            print(f"  {key}: {value}")
        if args.split_out:
            split_data = {
                'info': split_info,
                'train_indices': train_idx,
                'val_indices': val_idx,
                'test_indices': test_idx,
            }
            with open(args.split_out, 'w') as f:
                json.dump(split_data, f, indent=2)
            print(f"\nSplit saved to {args.split_out}")
        train_ds = Subset(full_ds, train_idx)
        val_ds = Subset(full_ds, val_idx)
    else:
        print("\nUsing random split...")
        train_len = int(args.train_frac * len(full_ds))
        val_len = int(args.val_frac * len(full_ds))
        test_len = len(full_ds) - train_len - val_len
        from torch.utils.data import random_split
        train_ds, val_ds, _ = random_split(
            full_ds,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(args.seed)
        )
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {test_len}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_pad, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_pad, num_workers=0
    )

    if args.domain_split_idx == 0:
        print("\nMode: STANDARD RNA PRETRAINING (domain isolation OFF)")
    else:
        print(f"\nMode: CHIMERIC RNA (domain split at {args.domain_split_idx})")

    print("\nInitializing model...")
    config = Config()
    config.RESNET_LAYERS = args.resnet_layers
    config.HIDDEN_DIM = args.hidden_dim
    config.LSTM_HIDDEN = args.lstm_hidden
    config.DEVICE = args.device
    config.DEFAULT_TRNA_LEN = args.domain_split_idx

    model = SpotRNA_LSTM_Refined_BPPM_Chimeric(config).to(args.device)

    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"Loading pretrained weights from {args.pretrained_path}")
        try:
            state_dict = torch.load(args.pretrained_path, map_location=args.device)
            model.load_state_dict(state_dict, strict=False)
            print("Weights loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load weights: {e}")

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    criterion = ChimericRNALoss(
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
        w_dice=args.w_dice,
        lambda_cross=args.lambda_cross,
        lambda_diag=args.lambda_diag,
        pos_weight=args.pos_weight
    )

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}, Gradient accumulation: {args.accum_steps}")
    print(f"Domain split (tRNA length): {args.domain_split_idx}")
    print(f"Cross-domain penalty lambda_cross: {args.lambda_cross}")
    print(f"Learning rate: {args.lr}, Pos weight: {args.pos_weight}")

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_f1 = 0.0
    nan_count = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        valid_batch_count = 0

        for batch_idx, (seqs, bppms, labels, masks) in enumerate(train_loader):
            seqs = seqs.to(args.device)
            bppms = bppms.to(args.device)
            labels = labels.to(args.device)
            masks = masks.to(args.device)
            B, L, _ = seqs.shape

            logits = model(
                seqs,
                bppm=bppms,
                mask=masks,
                domain_split_idx=args.domain_split_idx
            )

            seg_ids = get_seg_ids(B, L, args.domain_split_idx, args.device)
            debug_flag = (epoch == 0 and batch_idx == 0)
            loss = criterion(logits, labels, masks, seg_ids=seg_ids, debug=debug_flag)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if nan_count <= 5:
                    print(f"WARNING: Batch {batch_idx} NaN/Inf loss (count={nan_count}). Skipping.")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss = loss / args.accum_steps
            loss.backward()

            has_nan_grad = False
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break

            if has_nan_grad:
                nan_count += 1
                if nan_count <= 5:
                    print(f"WARNING: NaN gradient at batch {batch_idx} (count={nan_count}).")
                optimizer.zero_grad(set_to_none=True)
                continue

            current_real_loss = loss.item() * args.accum_steps
            total_loss += current_real_loss
            valid_batch_count += 1

            if (batch_idx + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                has_nan_param = any(torch.isnan(p).any() for p in model.parameters())
                if has_nan_param:
                    print(f"CRITICAL: NaN parameter after step at batch {batch_idx}.")
                    torch.save({
                        'epoch': epoch,
                        'batch': batch_idx,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }, os.path.join(args.save_dir, "debug_nan_checkpoint.pth"))
                    raise RuntimeError("NaN parameter detected. Training halted.")

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{batch_idx}] Loss: {current_real_loss:.4f} (skipped: {nan_count})")

        if (batch_idx + 1) % args.accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / max(valid_batch_count, 1)
        print(f"=== Epoch {epoch + 1} finished, Avg Loss: {avg_loss:.4f} "
              f"(valid: {valid_batch_count}/{len(train_loader)}, NaN skipped: {nan_count}) ===")

        model.eval()
        val_loss = 0.0
        val_f1 = 0.0
        val_p = 0.0
        val_r = 0.0

        with torch.no_grad():
            for batch_idx_val, (seqs, bppms, labels, masks) in enumerate(val_loader):
                seqs = seqs.to(args.device)
                bppms = bppms.to(args.device)
                labels = labels.to(args.device)
                masks = masks.to(args.device)
                B, L, _ = seqs.shape

                logits = model(
                    seqs,
                    bppm=bppms,
                    mask=masks,
                    domain_split_idx=args.domain_split_idx
                )

                seg_ids = get_seg_ids(B, L, args.domain_split_idx, args.device)
                loss = criterion(logits, labels, masks, seg_ids=seg_ids)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs > args.pp_offset).float()

                if masks.dim() == 2:
                    mask_2d = masks.unsqueeze(2) * masks.unsqueeze(1)
                else:
                    mask_2d = masks
                mask_2d = mask_2d.view_as(preds)

                preds = preds * mask_2d
                labels_masked = labels * mask_2d

                device_local = logits.device
                tril_mask = torch.tril(torch.ones(L, L, device=device_local), diagonal=-1).bool().unsqueeze(0)
                preds = preds * tril_mask
                labels_masked = labels_masked * tril_mask

                TP = (preds * labels_masked).sum().item()
                FP = (preds * (1 - labels_masked)).sum().item()
                FN = ((1 - preds) * labels_masked).sum().item()

                p = TP / (TP + FP + 1e-8)
                r = TP / (TP + FN + 1e-8)
                f1 = 2 * p * r / (p + r + 1e-8)

                val_f1 += f1
                val_p += p
                val_r += r

                if batch_idx_val == 0:
                    print("\n" + "="*20 + " 透视镜 " + "="*20)
                    print(f"当前 Batch 最高预测概率: {probs.max().item():.4f}")
                    print(f"当前 Batch 最低预测概率: {probs.min().item():.4f}")
                    print(f"当前 Batch 平均预测概率: {probs.mean().item():.4f}")
                    print(f"当前 Batch 的 Raw F1: {f1:.4f} (P: {p:.4f}, R: {r:.4f})")
                    cross_pred = (preds > 0) & (seg_ids.unsqueeze(2) != seg_ids.unsqueeze(1))
                    print(f"跨域假阳性配对数: {cross_pred.sum().item()} (应为0)")
                    print("="*52 + "\n")

        avg_val_loss = val_loss / max(len(val_loader), 1)
        avg_val_f1 = val_f1 / max(len(val_loader), 1)
        avg_val_p = val_p / max(len(val_loader), 1)
        avg_val_r = val_r / max(len(val_loader), 1)

        print(
            f"=== Validation Loss: {avg_val_loss:.4f} | "
            f"F1: {avg_val_f1:.4f} | P: {avg_val_p:.4f} R: {avg_val_r:.4f} | "
            f"pp_offset={args.pp_offset} min_loop={args.pp_min_loop} ===\n"
        )

        save_path = os.path.join(args.save_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)

        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            best_path = os.path.join(args.save_dir, "model_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved with F1: {best_val_f1:.4f}")

    print(f"\nTraining completed! Best validation F1: {best_val_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Train RNA structure prediction model (Chimeric RNA)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--max_len', type=int, default=600)
    parser.add_argument('--clstr_path', type=str, default=None)
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--split_out', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--accum_steps', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--pos_weight', type=float, default=3.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resnet_layers', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lstm_hidden', type=int, default=64)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--pp_offset', type=float, default=0.5)
    parser.add_argument('--pp_min_loop', type=int, default=4)
    parser.add_argument('--domain_split_idx', type=int, default=76)
    parser.add_argument('--lambda_cross', type=float, default=5.0)
    parser.add_argument('--lambda_diag', type=float, default=1.0)
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--w_dice', type=float, default=0.3)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_str = args.device
        try:
            if device_str.startswith('cuda'):
                if not torch.cuda.is_available():
                    raise ValueError("CUDA requested but not available")
                if ':' in device_str:
                    parts = device_str.split(':')
                    if len(parts) != 2 or not parts[1].isdigit():
                        raise ValueError(f"Invalid CUDA device format: '{device_str}'")
                    device_idx = int(parts[1])
                    if device_idx >= torch.cuda.device_count():
                        raise ValueError(f"CUDA device {device_idx} not available")
            args.device = torch.device(device_str)
        except (ValueError, RuntimeError) as e:
            print(f"Error: Invalid device '{device_str}': {e}")
            sys.exit(1)

    train_model(args)


if __name__ == '__main__':
    main()
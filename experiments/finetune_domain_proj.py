#!/usr/bin/env python3
"""
Domain-Proj 轻量 Fine-tune
============================
只训练 domain_proj + seg_pos_emb，让模型学会处理三段式嵌合体。

用法:
    python experiments/finetune_domain_proj.py \
        --pretrained checkpoints_v2_tr0_bppm_mfe/model_best.pth \
        --chimera_path data/pseudo_chimera.pkl \
        --save_dir checkpoints_v2_chimera_ft \
        --epochs 10 --device cuda:0

    python experiments/finetune_domain_proj.py \
        --pretrained checkpoints_v2_tr0_bppm_mfe/model_best.pth \
        --chimera_path data/pseudo_chimera.pkl \
        --save_dir checkpoints_v2_chimera_full_ft \
        --epochs 5 --lr 1e-5 --unfreeze_all \
        --device cuda:0
"""

import os, sys, pickle, random, argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.model import SpotRNA_LSTM_Refined_BPPM_Chimeric
from collections import namedtuple

_ChimeraWrap = namedtuple('_ChimeraWrap', 'lengths sequence contact_map domain_labels')

def _wrap_chimera(c):
    """兼容 dict 和 ChimeraSample 对象"""
    if hasattr(c, 'lengths'):
        return c
    return _ChimeraWrap(
        lengths=c['lengths'],
        sequence=c['sequence'],
        contact_map=np.array(c['contact_map']) if not isinstance(c.get('contact_map'), np.ndarray) else c['contact_map'],
        domain_labels=np.array(c['domain_labels']) if not isinstance(c.get('domain_labels'), np.ndarray) else c['domain_labels'],
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        Chimera Dataset                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ChimeraDataset(Dataset):
    """包装伪嵌合体数据为训练格式"""
    def __init__(self, chimera_path, max_len=300):
        with open(chimera_path, 'rb') as f:
            raw = pickle.load(f)

        self.samples = []
        for c in raw:
            if isinstance(c, dict):
                L = c['lengths']['total']
            else:
                L = c.lengths['total']
            if L > max_len:
                continue
            self.samples.append(c)

        print(f"[chimera] {len(self.samples)} 条 ≤ {max_len}nt (总 {len(raw)})")

    @staticmethod
    def _info(c):
        if isinstance(c, dict):
            return (c['lengths']['total'], c['lengths']['trna_5'], c['lengths']['trna_3'],
                    c['sequence'],
                    np.array(c['contact_map']) if not isinstance(c['contact_map'], np.ndarray) else c['contact_map'],
                    np.array(c['domain_labels']) if not isinstance(c['domain_labels'], np.ndarray) else c['domain_labels'])
        else:
            return (c.lengths['total'], c.lengths['trna_5'], c.lengths['trna_3'],
                    c.sequence, c.contact_map, c.domain_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        c = self.samples[idx]
        L, t5, t3, seq, cmap, dom = self._info(c)

        m = {'A':0,'C':1,'G':2,'U':3,'T':3}
        oh = np.zeros((L,4), dtype=np.float32)
        for i,ch in enumerate(seq[:L]):
            if ch.upper() in m:
                oh[i, m[ch.upper()]] = 1.0

        cmap = cmap[:L,:L].copy().astype(np.float32)
        dom = dom[:L].copy()

        return (torch.tensor(oh), torch.tensor(cmap), torch.tensor(dom, dtype=torch.long),
                t5, t3)

def chimera_collate(batch):
    max_len = max(b[0].shape[0] for b in batch)
    B = len(batch)

    seqs = torch.zeros(B, max_len, 4)
    bppms = torch.zeros(B, max_len, max_len)
    labels = torch.zeros(B, max_len, max_len)
    masks = torch.zeros(B, max_len)
    t5s = torch.zeros(B, dtype=torch.long)
    t3s = torch.zeros(B, dtype=torch.long)

    for i, (oh, cmap, domain, t5, t3) in enumerate(batch):
        n = oh.shape[0]
        seqs[i,:n] = oh
        labels[i,:n,:n] = cmap
        masks[i,:n] = 1.0
        t5s[i] = t5
        t3s[i] = t3

    return seqs, bppms, labels, masks, t5s, t3s


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        Loss                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ChimeraLoss(nn.Module):
    """嵌合体训练专用 BCE + Dice"""
    def __init__(self, pos_weight=5.0, w_dice=0.2):
        super().__init__()
        self.pos_weight = pos_weight
        self.w_dice = w_dice

    def forward(self, logits, targets, valid_mask):
        B = logits.shape[0]

        weight = torch.where(targets > 0.5,
                             torch.full_like(logits, self.pos_weight),
                             torch.ones_like(logits))
        bce = F.binary_cross_entropy_with_logits(logits, targets, weight=weight,
                                                   reduction='none')
        bce = (bce * valid_mask).sum() / valid_mask.sum().clamp(min=1)

        if self.w_dice > 0:
            prob = torch.sigmoid(logits) * valid_mask
            inter = (prob * targets * valid_mask).sum(dim=(-2,-1))
            union = (prob * valid_mask).sum(dim=(-2,-1)) + (targets * valid_mask).sum(dim=(-2,-1))
            dice = (1 - (2*inter + 1e-6) / (union + 1e-6)).mean()
        else:
            dice = torch.tensor(0.0, device=logits.device)

        return bce + self.w_dice * dice, {'bce': bce.item(), 'dice': dice.item()}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        Train                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--chimera_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--unfreeze_all', action='store_true',
                        help='解冻全部参数 (默认只训 domain_proj + seg_pos_emb)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── 模型 ──
    class Cfg: HIDDEN_DIM=64; RESNET_LAYERS=8; LSTM_HIDDEN=64
    model = SpotRNA_LSTM_Refined_BPPM_Chimeric(Cfg).to(device)
    sd = torch.load(args.pretrained, map_location=device)
    if 'model_state_dict' in sd: sd = sd['model_state_dict']
    model.load_state_dict(sd, strict=False)
    print(f"[model] loaded {args.pretrained}")

    # ── 冻结策略 ──
    if args.unfreeze_all:
        for p in model.parameters():
            p.requires_grad = True
        print("[mode] Full fine-tune (all params)")
    else:
        for name, p in model.named_parameters():
            p.requires_grad = ('domain_proj' in name or 'seg_pos_emb' in name)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[mode] Domain-only fine-tune ({n_train:,}/{n_total:,} params, {100*n_train/n_total:.1f}%)")
        trainable = [n for n,p in model.named_parameters() if p.requires_grad]
        for n in trainable:
            print(f"  ✓ {n}")

    # ── 数据 ──
    ds = ChimeraDataset(args.chimera_path)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=chimera_collate, num_workers=0)
    print(f"[data] {len(ds)} chimera samples, {len(loader)} batches/epoch")

    # ── 优化器 ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                            eta_min=args.lr*0.1)
    criterion = ChimeraLoss(pos_weight=5.0, w_dice=0.2)

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = total_bce = total_dice = 0.0

        for step, (seqs, bppms, labels, masks, t5s, t3s) in enumerate(loader):
            seqs = seqs.to(device)
            bppms = bppms.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            B, L = seqs.shape[0], seqs.shape[1]

            logits = model(seqs, bppm=bppms, mask=masks,
                           trna_5end_len=t5s, trna_3end_len=t3s)

            # valid_mask: 排除 padding + 对角线 + 跨域
            m1d = masks
            pad_2d = m1d.unsqueeze(2) * m1d.unsqueeze(1)
            diag = torch.eye(L, device=device).unsqueeze(0)
            valid = pad_2d * (1 - diag)
            # 跨域遮蔽
            seg_ids = torch.zeros(B, L, dtype=torch.long, device=device)
            for b in range(B):
                t5 = t5s[b].item()
                t3 = t3s[b].item()
                if t5 + t3 < L:
                    seg_ids[b, t5:L-t3] = 2
            same_dom = (seg_ids.unsqueeze(2) == seg_ids.unsqueeze(1)).float()
            valid = valid * same_dom

            loss, comps = criterion(logits, labels, valid)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += comps['bce'] + comps['dice']
            total_bce += comps['bce']
            total_dice += comps['dice']

            if step % 50 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} step {step:4d} "
                      f"BCE={comps['bce']:.4f} Dice={comps['dice']:.4f}")

        scheduler.step()
        n = max(1, len(loader))
        avg_loss = total_loss / n
        print(f"\n  Epoch {epoch+1} avg BCE={total_bce/n:.4f} Dice={total_dice/n:.4f} "
              f"LR={scheduler.get_last_lr()[0]:.2e}")

        # 保存
        ckpt = os.path.join(args.save_dir, f"model_epoch_{epoch+1:03d}.pth")
        torch.save(model.state_dict(), ckpt)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.save_dir, "model_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  ★ best ({best_loss:.4f})")

    print(f"\nDone. Best model: {best_path}")


if __name__ == '__main__':
    main()

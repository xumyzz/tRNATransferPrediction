#!/usr/bin/env python3
"""
Enhanced training script with support for cluster-based data splitting.

This script extends the training workflow to support:
- Command-line arguments for all hyperparameters
- Cluster-based train/val/test splits using cd-hit-est output
- Export of split indices for reproducibility

Usage:
    # Standard random split
    python train_with_args.py --data_dir data/TR0 --max_len 300 --epochs 10
    
    # Cluster-based split
    python train_with_args.py --data_dir data/bpRNA_1m --max_len 600 \
        --clstr_path clusters.clstr --train_frac 0.8 --val_frac 0.1 \
        --split_out splits.json
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import os
import argparse
import json
import random
import numpy as np

from src.config import Config
from src.utils import compute_masked_loss, calculate_f1
from src.dataset import MultiFileDatasetUpgrade, collate_pad
from src.model import SpotRNA_LSTM_Refined
from scripts.cluster_utils import parse_cd_hit_clusters


def create_cluster_split(dataset, clstr_path, train_frac, val_frac, seed=42):
    """
    Create train/val/test split based on clusters.
    
    Ensures sequences from the same cluster stay together in the same split.
    
    Args:
        dataset: Dataset with .names attribute
        clstr_path: Path to cd-hit-est .clstr output
        train_frac: Fraction of clusters for training
        val_frac: Fraction of clusters for validation
        seed: Random seed for reproducibility
    
    Returns:
        (train_indices, val_indices, test_indices), split_info
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Parse clusters
    clusters = parse_cd_hit_clusters(clstr_path)
    print(f"Found {len(clusters)} clusters")
    
    # Create name to index mapping
    name_to_idx = {name: idx for idx, name in enumerate(dataset.names)}
    
    # Map cluster names to dataset indices
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
    
    # Shuffle clusters
    random.shuffle(cluster_indices)
    
    # Split clusters
    n_clusters = len(cluster_indices)
    n_train = int(n_clusters * train_frac)
    n_val = int(n_clusters * val_frac)
    
    train_clusters = cluster_indices[:n_train]
    val_clusters = cluster_indices[n_train:n_train + n_val]
    test_clusters = cluster_indices[n_train + n_val:]
    
    # Flatten to get indices
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
    """Main training function"""
    print(f"Using device: {args.device}")
    print(f"Configuration: {vars(args)}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    full_ds = MultiFileDatasetUpgrade(args.data_dir, max_len=args.max_len)
    
    if len(full_ds) == 0:
        print("Error: No data loaded. Check path.")
        return
    
    print(f"Loaded {len(full_ds)} sequences")
    
    # Create data splits
    if args.clstr_path:
        print(f"\nCreating cluster-based split from {args.clstr_path}...")
        (train_idx, val_idx, test_idx), split_info = create_cluster_split(
            full_ds, args.clstr_path, args.train_frac, args.val_frac, args.seed
        )
        
        print("\nSplit information:")
        for key, value in split_info.items():
            print(f"  {key}: {value}")
        
        # Save split info if requested
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
        # Random split
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
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_pad,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_pad,
        num_workers=0
    )
    
    # Initialize model
    print("\nInitializing model...")
    # Create a config object with args
    config = Config()
    config.RESNET_LAYERS = args.resnet_layers
    config.HIDDEN_DIM = args.hidden_dim
    config.LSTM_HIDDEN = args.lstm_hidden
    config.DEVICE = args.device
    
    model = SpotRNA_LSTM_Refined(config).to(args.device)
    
    # Load pretrained weights if specified
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"Loading pretrained weights from {args.pretrained_path}")
        try:
            state_dict = torch.load(args.pretrained_path, map_location=args.device)
            model.load_state_dict(state_dict)
            print("Weights loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load weights: {e}")
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}, Gradient accumulation: {args.accum_steps}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_f1 = 0.0
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        
        for batch_idx, (seqs, labels, masks) in enumerate(train_loader):
            seqs = seqs.to(args.device)
            labels = labels.to(args.device)
            masks = masks.to(args.device)
            
            logits = model(seqs, mask=masks)
            loss = compute_masked_loss(logits, labels, masks, pos_weight=args.pos_weight)
            
            loss = loss / args.accum_steps
            loss.backward()
            
            current_real_loss = loss.item() * args.accum_steps
            total_loss += current_real_loss
            
            if (batch_idx + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()
            
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{batch_idx}] Loss: {current_real_loss:.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"=== Epoch {epoch + 1} finished, Avg Loss: {avg_loss:.4f} ===")
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_f1 = 0
        
        with torch.no_grad():
            for seqs, labels, masks in val_loader:
                seqs = seqs.to(args.device)
                labels = labels.to(args.device)
                masks = masks.to(args.device)
                
                logits = model(seqs, mask=masks)
                loss = compute_masked_loss(logits, labels, masks, pos_weight=args.pos_weight)
                
                val_loss += loss.item()
                f1 = calculate_f1(logits, labels, masks)
                val_f1 += f1
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_f1 = val_f1 / len(val_loader)
        
        print(f"=== Validation Loss: {avg_val_loss:.4f} | F1: {avg_val_f1:.4f} ===\n")
        
        # Save checkpoint
        save_path = os.path.join(args.save_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)
        
        # Save best model
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            best_path = os.path.join(args.save_dir, "model_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved with F1: {best_val_f1:.4f}")
    
    print(f"\nTraining completed! Best validation F1: {best_val_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Train RNA structure prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory or file')
    parser.add_argument('--max_len', type=int, default=600,
                        help='Maximum sequence length')
    
    # Cluster split arguments
    parser.add_argument('--clstr_path', type=str, default=None,
                        help='Path to cd-hit-est .clstr file for cluster-based splitting')
    parser.add_argument('--train_frac', type=float, default=0.8,
                        help='Fraction of data/clusters for training')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Fraction of data/clusters for validation')
    parser.add_argument('--split_out', type=str, default=None,
                        help='Output JSON file for split indices')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--accum_steps', type=int, default=32,
                        help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--pos_weight', type=float, default=3.5,
                        help='Positive class weight for BCE loss')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Model arguments
    parser.add_argument('--resnet_layers', type=int, default=8,
                        help='Number of ResNet layers')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--lstm_hidden', type=int, default=64,
                        help='LSTM hidden dimension')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained model weights')
    
    # Other arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu), auto-detected if not specified')
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(args.device)
    
    train_model(args)


if __name__ == '__main__':
    main()

# main.py
import argparse
from src.train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train tRNA structure prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with random split (default)
  python main.py
  
  # Train with cluster-based split to prevent leakage
  python main.py --clstr_path /path/to/tr0_cdhit95.clstr --split_out splits.json
  
  # Customize split fractions
  python main.py --clstr_path /path/to/tr0_cdhit95.clstr --train_frac 0.8 --val_frac 0.1
        """
    )
    
    parser.add_argument(
        '--clstr_path',
        type=str,
        default=None,
        help='Path to CD-HIT .clstr file for cluster-based splitting'
    )
    parser.add_argument(
        '--split_seed',
        type=int,
        default=42,
        help='Random seed for split (default: 42)'
    )
    parser.add_argument(
        '--train_frac',
        type=float,
        default=0.8,
        help='Fraction of clusters for training (default: 0.8)'
    )
    parser.add_argument(
        '--val_frac',
        type=float,
        default=0.1,
        help='Fraction of clusters for validation (default: 0.1)'
    )
    parser.add_argument(
        '--split_out',
        type=str,
        default=None,
        help='Path to save split configuration JSON (optional)'
    )
    
    args = parser.parse_args()
    
    # Call train with cluster split parameters
    train(
        clstr_path=args.clstr_path,
        split_seed=args.split_seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        split_out=args.split_out
    )
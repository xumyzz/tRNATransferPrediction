"""Main entry point: use CLI args when provided, else Config defaults."""
import sys

from src.train import train, main as train_main

if __name__ == '__main__':
    # Use CLI args when provided, else fall back to Config defaults.
    if len(sys.argv) > 1:
        train_main()
    else:
        train()
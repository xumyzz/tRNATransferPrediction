# main.py
import sys

from src.train import train, main as train_main

if __name__ == '__main__':
    if len(sys.argv) > 1:
        train_main()
    else:
        train()
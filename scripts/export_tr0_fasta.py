#!/usr/bin/env python3
"""
Export TR0 dataset to FASTA format for CD-HIT clustering.

This script applies the same validation and filtering as training:
- Max length filtering (default 300)
- N proportion threshold
- Sequence/structure length match
- Base validation and optional pseudoknot filtering
"""

import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts import export_bprna_fasta


def main():
    parser = argparse.ArgumentParser(
        description='Export TR0 dataset to FASTA format for clustering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', required=True,
                        help='Directory containing .st files (TR0 dataset)')
    parser.add_argument('--max_len', type=int, default=300,
                        help='Maximum sequence length')
    parser.add_argument('--n_threshold', type=float, default=0.2,
                        help='Maximum proportion of N bases allowed')
    parser.add_argument('--allow_pseudoknot', action='store_true',
                        help='Allow structures with pseudoknot notation ([], {}, <>). Default is to filter them out.')
    parser.add_argument('--out_fasta', required=True,
                        help='Output FASTA file')
    parser.add_argument('--out_names', default=None,
                        help='Output file for names list (one per line)')
    parser.add_argument('--stats_out', default=None,
                        help='Output JSON file for statistics')

    args = parser.parse_args()
    export_bprna_fasta.run_export(
        inputs=[args.data_dir],
        max_len=args.max_len,
        n_threshold=args.n_threshold,
        allow_pseudoknot=args.allow_pseudoknot,
        out_fasta=args.out_fasta,
        out_names=args.out_names,
        stats_out_path=args.stats_out
    )


if __name__ == '__main__':
    main()

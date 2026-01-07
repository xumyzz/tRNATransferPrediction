#!/usr/bin/env python3
"""
Export bpRNA dataset to FASTA format for clustering with cd-hit-est.

This script reads .st and .dbn files from specified inputs (files or directories),
validates sequences, and exports them to FASTA format with names matching the
dataset's name field.

Usage:
    python scripts/export_bprna_fasta.py \
        --input data/bpRNA_CRW \
        --input data/bpRNA_RFAM \
        --max_len 600 \
        --out_fasta bprna_1m.fasta \
        --out_names bprna_1m_names.txt \
        --stats_out bprna_1m_stats.json
"""

import os
import sys
import argparse
import glob
import json


def parse_st_file(fpath):
    """Parse bpRNA .st format files and yield (name, seq, struct) tuples"""
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.rstrip() for line in f]
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for #Name: header
        if line.startswith("#Name:"):
            name = line[6:].strip()
            
            # Skip other comment lines
            i += 1
            while i < len(lines) and lines[i].strip().startswith("#"):
                i += 1
            
            # Next should be sequence
            if i >= len(lines):
                break
            seq_line = lines[i].strip()
            
            # Next should be structure
            i += 1
            if i >= len(lines):
                break
            struct_line = lines[i].strip()
            
            yield (name, seq_line, struct_line)
        i += 1


def parse_dbn_file(fpath):
    """Parse .dbn format files and yield (name, seq, struct) tuples"""
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.rstrip() for line in f]
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Look for header line starting with >
        if line.startswith(">"):
            name = line[1:].strip()
            if not name:
                # Use a default name with line number
                name = f"seq_{i+1}"
            
            # Next non-empty line should be sequence
            i += 1
            seq_line = ""
            while i < len(lines):
                line = lines[i].strip()
                if line:
                    seq_line = line
                    break
                i += 1
            
            if not seq_line:
                break
            
            # Next non-empty line should be structure
            i += 1
            struct_line = ""
            while i < len(lines):
                line = lines[i].strip()
                if line:
                    struct_line = line
                    break
                i += 1
            
            if not struct_line:
                break
            
            yield (name, seq_line, struct_line)
        i += 1


def is_valid_entry(name, seq, struct, max_len, n_threshold, stats):
    """Validate an entry and update stats"""
    stats["total"] += 1
    
    # Normalize sequence
    seq = seq.upper().replace('T', 'U')
    
    # Length check
    if len(seq) > max_len:
        stats["too_long"] += 1
        return False, None, None
    
    # Length match check
    if len(seq) != len(struct):
        stats["length_mismatch"] += 1
        return False, None, None
    
    # Validate sequence only contains valid bases
    valid_bases = set('ACGUN')
    if not all(c in valid_bases for c in seq):
        stats["invalid_bases"] += 1
        return False, None, None
    
    # N threshold check
    if len(seq) > 0 and seq.count('N') / len(seq) > n_threshold:
        stats["too_many_n"] += 1
        return False, None, None
    
    stats["kept"] += 1
    return True, seq, struct


def collect_files(inputs):
    """Collect all .st and .dbn files from input paths"""
    files = []
    for input_path in inputs:
        if os.path.isfile(input_path):
            files.append(input_path)
        elif os.path.isdir(input_path):
            # Collect .st and .dbn files from directory
            st_files = glob.glob(os.path.join(input_path, "*.st"))
            dbn_files = glob.glob(os.path.join(input_path, "*.dbn"))
            files.extend(st_files + dbn_files)
        else:
            print(f"Warning: {input_path} is not a valid file or directory")
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description='Export bpRNA dataset to FASTA format for clustering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', action='append', required=True,
                        help='Input file or directory (can be specified multiple times)')
    parser.add_argument('--max_len', type=int, default=600,
                        help='Maximum sequence length')
    parser.add_argument('--n_threshold', type=float, default=0.2,
                        help='Maximum proportion of N bases allowed')
    parser.add_argument('--out_fasta', required=True,
                        help='Output FASTA file')
    parser.add_argument('--out_names', default=None,
                        help='Output file for names list (one per line)')
    parser.add_argument('--stats_out', default=None,
                        help='Output JSON file for statistics')
    
    args = parser.parse_args()
    
    # Collect all files
    files = collect_files(args.input)
    print(f"Found {len(files)} files to process")
    
    # Initialize stats
    stats = {
        "files_read": 0,
        "total": 0,
        "kept": 0,
        "too_long": 0,
        "length_mismatch": 0,
        "too_many_n": 0,
        "invalid_bases": 0,
        "parse_errors": 0
    }
    
    # Process files and write FASTA
    names_list = []
    with open(args.out_fasta, 'w') as fasta_out:
        for fpath in files:
            try:
                stats["files_read"] += 1
                
                # Determine file type and parse
                if fpath.endswith('.st'):
                    entries = parse_st_file(fpath)
                elif fpath.endswith('.dbn'):
                    entries = parse_dbn_file(fpath)
                else:
                    continue
                
                # Process entries
                for name, seq, struct in entries:
                    valid, normalized_seq, _ = is_valid_entry(
                        name, seq, struct, args.max_len, args.n_threshold, stats
                    )
                    
                    if valid:
                        # Write to FASTA
                        fasta_out.write(f">{name}\n")
                        fasta_out.write(f"{normalized_seq}\n")
                        names_list.append(name)
                        
            except Exception as e:
                print(f"Error processing {os.path.basename(fpath)}: {e}")
                stats["parse_errors"] += 1
    
    # Write names file if requested
    if args.out_names:
        with open(args.out_names, 'w') as names_out:
            for name in names_list:
                names_out.write(f"{name}\n")
    
    # Write stats if requested
    if args.stats_out:
        with open(args.stats_out, 'w') as stats_out:
            json.dump(stats, stats_out, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Export Summary")
    print("=" * 50)
    print(f"Files read: {stats['files_read']}")
    print(f"Total records: {stats['total']}")
    print(f"Records kept: {stats['kept']}")
    print(f"Filtered out:")
    print(f"  - Too long: {stats['too_long']}")
    print(f"  - Length mismatch: {stats['length_mismatch']}")
    print(f"  - Too many Ns: {stats['too_many_n']}")
    print(f"  - Invalid bases: {stats['invalid_bases']}")
    print(f"  - Parse errors: {stats['parse_errors']}")
    print("=" * 50)
    print(f"\nFASTA written to: {args.out_fasta}")
    if args.out_names:
        print(f"Names written to: {args.out_names}")
    if args.stats_out:
        print(f"Stats written to: {args.stats_out}")


if __name__ == '__main__':
    main()

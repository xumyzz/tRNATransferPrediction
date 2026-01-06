#!/usr/bin/env python3
"""
Export FASTA from TR0 .st files for CD-HIT clustering.

This script parses bpRNA-format .st files and exports sequences that pass
validation and length filtering (same criteria as training) to a FASTA file.
"""

import argparse
import glob
import os
import sys


def parse_bprna_files(data_dir, max_len=300, max_n_ratio=0.2):
    """
    Parse bpRNA .st files and extract valid sequences.
    
    Args:
        data_dir: Directory containing .st files
        max_len: Maximum sequence length (default: 300)
        max_n_ratio: Maximum ratio of N nucleotides allowed (default: 0.2)
    
    Returns:
        List of tuples (name, sequence) for valid entries
    """
    # Get file list
    if os.path.isfile(data_dir):
        file_list = [data_dir]
    else:
        file_list = sorted(glob.glob(os.path.join(data_dir, "*.st")))
        if not file_list:
            file_list = sorted(glob.glob(os.path.join(data_dir, "*.dbn")))
    
    print(f"🧐 正在扫描 {len(file_list)} 个文件 (MaxLen={max_len})...")
    
    # Statistics
    stats = {"total": 0, "kept": 0, "long": 0, "error": 0}
    valid_entries = []
    
    for fpath in file_list:
        try:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            # State machine parser
            # state 0: looking for Name
            # state 1: looking for Seq
            # state 2: looking for Struct
            current_entry = {}
            state = 0
            
            for line in lines:
                # 1. New entry starts with #Name: or >
                if line.startswith("#Name:") or line.startswith(">"):
                    # Save previous entry if complete
                    if state == 2 and 'name' in current_entry and 'seq' in current_entry and 'struct' in current_entry:
                        if is_valid_entry(current_entry, max_len, stats, max_n_ratio):
                            valid_entries.append((current_entry['name'], current_entry['seq']))
                    
                    # Reset for new entry
                    current_entry = {}
                    # Extract name
                    if line.startswith("#Name:"):
                        current_entry['name'] = line.split(":", 1)[1].strip()
                    else:  # starts with ">"
                        current_entry['name'] = line[1:].split()[0] if len(line) > 1 else "unknown"
                    state = 1
                    continue
                
                # 2. Skip other comment lines
                if line.startswith("#"):
                    continue
                
                # 3. Looking for sequence (state 1)
                if state == 1:
                    # If line contains brackets, it's a structure line (error)
                    if any(c in "().[]{}<>" for c in line):
                        state = 0
                        continue
                    
                    # Normalize sequence: uppercase, T->U
                    current_entry['seq'] = line.upper().replace('T', 'U')
                    state = 2
                    continue
                
                # 4. Looking for structure (state 2)
                if state == 2:
                    if any(c in "().[]{}<>" for c in line):
                        current_entry['struct'] = line
                        # Validate and save
                        if is_valid_entry(current_entry, max_len, stats, max_n_ratio):
                            valid_entries.append((current_entry['name'], current_entry['seq']))
                        current_entry = {}
                        state = 0
                    else:
                        state = 0
            
            # Don't forget last entry
            if 'name' in current_entry and 'seq' in current_entry and 'struct' in current_entry:
                if is_valid_entry(current_entry, max_len, stats, max_n_ratio):
                    valid_entries.append((current_entry['name'], current_entry['seq']))
        
        except Exception as e:
            print(f"⚠️ 读取 {os.path.basename(fpath)} 失败: {e}")
    
    print("\n" + "=" * 30)
    print(f"📊 解析报告 (MaxLen={max_len})")
    print(f"✅ 有效序列: {stats['kept']}")
    print(f"❌ 超长丢弃: {stats['long']}")
    print(f"❌ 格式/N多: {stats['error']}")
    print("=" * 30 + "\n")
    
    return valid_entries


def is_valid_entry(entry, max_len, stats, max_n_ratio=0.2):
    """
    Validate an entry using same criteria as training.
    
    Args:
        entry: Dict with 'name', 'seq', 'struct' keys
        max_len: Maximum length threshold
        stats: Dict to update statistics
        max_n_ratio: Maximum ratio of N nucleotides allowed (default: 0.2)
    
    Returns:
        True if valid, False otherwise
    """
    seq = entry['seq']
    struct = entry['struct']
    stats["total"] += 1
    
    # 1. Length check
    if len(seq) > max_len:
        stats["long"] += 1
        return False
    
    # 2. Length matching check
    if len(seq) != len(struct):
        stats["error"] += 1
        return False
    
    # 3. N content check (configurable threshold)
    if seq.count('N') / len(seq) > max_n_ratio:
        stats["error"] += 1
        return False
    
    # 4. Valid
    stats["kept"] += 1
    return True


def write_fasta(entries, out_fasta):
    """
    Write entries to FASTA file.
    
    Args:
        entries: List of (name, sequence) tuples
        out_fasta: Output FASTA file path
    """
    with open(out_fasta, 'w') as f:
        for name, seq in entries:
            f.write(f">{name}\n")
            f.write(f"{seq}\n")
    print(f"✅ 写入 {len(entries)} 条序列到 {out_fasta}")


def write_names_list(entries, out_names):
    """
    Write list of names to file (optional).
    
    Args:
        entries: List of (name, sequence) tuples
        out_names: Output names file path
    """
    with open(out_names, 'w') as f:
        for name, _ in entries:
            f.write(f"{name}\n")
    print(f"✅ 写入 {len(entries)} 个名称到 {out_names}")


def main():
    parser = argparse.ArgumentParser(
        description="Export FASTA from TR0 .st files for clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scripts/export_tr0_fasta.py \\
    --data_dir /path/to/TR0 \\
    --max_len 300 \\
    --out_fasta tr0_sequences.fasta \\
    --out_names tr0_names.txt
        """
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing .st files (or single .st file)'
    )
    parser.add_argument(
        '--max_len',
        type=int,
        default=300,
        help='Maximum sequence length (default: 300)'
    )
    parser.add_argument(
        '--max_n_ratio',
        type=float,
        default=0.2,
        help='Maximum ratio of N nucleotides allowed (default: 0.2)'
    )
    parser.add_argument(
        '--out_fasta',
        type=str,
        required=True,
        help='Output FASTA file path'
    )
    parser.add_argument(
        '--out_names',
        type=str,
        default=None,
        help='Optional output file for list of names'
    )
    
    args = parser.parse_args()
    
    # Check input directory/file exists
    if not os.path.exists(args.data_dir):
        print(f"❌ 错误: {args.data_dir} 不存在")
        sys.exit(1)
    
    # Parse files
    entries = parse_bprna_files(args.data_dir, args.max_len, args.max_n_ratio)
    
    if not entries:
        print("❌ 未找到有效序列")
        sys.exit(1)
    
    # Write FASTA
    write_fasta(entries, args.out_fasta)
    
    # Write names list if requested
    if args.out_names:
        write_names_list(entries, args.out_names)
    
    print("\n✅ 导出完成！")
    print(f"下一步:")
    print(f"  cd-hit-est -i {args.out_fasta} -o tr0_cdhit95 -c 0.95 -n 10 -d 0 -M 0 -T 0")


if __name__ == "__main__":
    main()

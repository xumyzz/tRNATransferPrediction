#!/usr/bin/env python3
"""
End-to-end workflow test demonstrating the bpRNA-1m pretraining pipeline.

This test simulates the complete workflow:
1. Create sample .st and .dbn data files
2. Export to FASTA
3. Simulate cluster file creation
4. Parse clusters for training split

Note: This test doesn't require torch/numpy and doesn't actually train a model.
"""

import os
import sys
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.cluster_utils import parse_cd_hit_clusters


def create_sample_data(data_dir):
    """Create sample .st and .dbn files"""
    
    # Create .st file (bpRNA format)
    st_file = os.path.join(data_dir, "sample_crw.st")
    with open(st_file, 'w') as f:
        f.write("#Name: bpRNA_CRW_001\n")
        f.write("#Length: 20\n")
        f.write("ACGUACGUACGUACGUACGU\n")
        f.write("((((....))))..((...)\n")
        f.write("#Name: bpRNA_CRW_002\n")
        f.write("#Length: 15\n")
        f.write("GGGGCCCCAAACCCC\n")
        f.write("(((())))((...)))\n")
    
    # Create .dbn file (FASTA-like format)
    dbn_file = os.path.join(data_dir, "sample_rfam.dbn")
    with open(dbn_file, 'w') as f:
        f.write(">bpRNA_RFAM_001\n")
        f.write("ACGUACGUACGU\n")
        f.write("(((...)))...\n")
        f.write("\n")
        f.write(">bpRNA_RFAM_002\n")
        f.write("GGCCGGCCGGCC\n")
        f.write("((((...))))\n")
        f.write("\n")
        f.write(">bpRNA_RFAM_003\n")
        f.write("AAAAUUUUCCCCGGGG\n")
        f.write("....((((....))))\n")
    
    return [st_file, dbn_file]


def workflow_test():
    """Test the complete workflow"""
    
    print("=" * 70)
    print("bpRNA-1m Pretraining Workflow Test")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n1. Creating sample data files...")
        data_dir = os.path.join(tmpdir, "data")
        os.makedirs(data_dir)
        
        files = create_sample_data(data_dir)
        print(f"   Created {len(files)} data files")
        print(f"   - {os.path.basename(files[0])} (.st format)")
        print(f"   - {os.path.basename(files[1])} (.dbn format)")
        
        print("\n2. Exporting to FASTA for clustering...")
        fasta_file = os.path.join(tmpdir, "sequences.fasta")
        names_file = os.path.join(tmpdir, "names.txt")
        stats_file = os.path.join(tmpdir, "stats.json")
        
        import subprocess
        result = subprocess.run([
            sys.executable,
            "scripts/export_bprna_fasta.py",
            "--input", data_dir,
            "--max_len", "600",
            "--out_fasta", fasta_file,
            "--out_names", names_file,
            "--stats_out", stats_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"   Error: {result.stderr}")
            return False
        
        # Read stats
        with open(stats_file) as f:
            stats = json.load(f)
        
        print(f"   Exported {stats['kept']} sequences to FASTA")
        
        # Read names
        with open(names_file) as f:
            names = [line.strip() for line in f]
        
        print(f"   Sequence names: {', '.join(names)}")
        
        print("\n3. Simulating CD-HIT clustering...")
        # In real workflow, user would run:
        # cd-hit-est -i sequences.fasta -o clustered.fasta -c 0.90 -n 8
        
        # Create a simulated cluster file
        clstr_file = os.path.join(tmpdir, "clustered.fasta.clstr")
        with open(clstr_file, 'w') as f:
            # Cluster 0: CRW sequences (similar)
            f.write(">Cluster 0\n")
            f.write("0\t20nt, >bpRNA_CRW_001... *\n")
            f.write("1\t15nt, >bpRNA_CRW_002... at 85%\n")
            # Cluster 1: RFAM sequences (similar to each other)
            f.write(">Cluster 1\n")
            f.write("0\t12nt, >bpRNA_RFAM_001... *\n")
            f.write("1\t12nt, >bpRNA_RFAM_002... at 95%\n")
            # Cluster 2: Unique RFAM sequence
            f.write(">Cluster 2\n")
            f.write("0\t16nt, >bpRNA_RFAM_003... *\n")
        
        clusters = parse_cd_hit_clusters(clstr_file)
        print(f"   Created {len(clusters)} clusters")
        for i, cluster in enumerate(clusters):
            print(f"   - Cluster {i}: {len(cluster)} sequences ({', '.join(cluster)})")
        
        print("\n4. Cluster-based split configuration...")
        total_clusters = len(clusters)
        train_frac = 0.6  # 60% of clusters for training
        val_frac = 0.2    # 20% for validation
        
        n_train = int(total_clusters * train_frac)
        n_val = int(total_clusters * val_frac)
        n_test = total_clusters - n_train - n_val
        
        print(f"   Total clusters: {total_clusters}")
        print(f"   Train clusters: {n_train} ({train_frac*100}%)")
        print(f"   Val clusters: {n_val} ({val_frac*100}%)")
        print(f"   Test clusters: {n_test} ({(1-train_frac-val_frac)*100}%)")
        
        print("\n5. Training command (example)...")
        cmd = f"""
python train_with_args.py \\
    --data_dir {data_dir} \\
    --max_len 600 \\
    --clstr_path {clstr_file} \\
    --train_frac {train_frac} \\
    --val_frac {val_frac} \\
    --split_out splits.json \\
    --batch_size 2 \\
    --accum_steps 32 \\
    --epochs 10 \\
    --save_dir checkpoints_bprna1m
        """.strip()
        
        print(f"   {cmd}")
        
        print("\n" + "=" * 70)
        print("✓ Workflow test completed successfully!")
        print("=" * 70)
        print("\nKey features demonstrated:")
        print("  1. ✓ Mixed .st and .dbn file support")
        print("  2. ✓ FASTA export with name tracking")
        print("  3. ✓ CD-HIT cluster file parsing")
        print("  4. ✓ Cluster-based split planning")
        print("  5. ✓ Training command generation")
        
        return True


if __name__ == '__main__':
    success = workflow_test()
    sys.exit(0 if success else 1)

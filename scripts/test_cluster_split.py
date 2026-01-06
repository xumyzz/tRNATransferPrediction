#!/usr/bin/env python3
"""
Test script to verify cluster-based splitting functionality.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import MultiFileDatasetUpgrade
from src.cluster_split import parse_cdhit_clstr, create_cluster_splits


def test_cluster_split():
    """Test cluster-based splitting with TR0 data."""
    
    print("=" * 60)
    print("Testing Cluster-Based Splitting")
    print("=" * 60)
    
    # Load a small subset of data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'TR0')
    
    print(f"\n1. Loading dataset from {data_dir}...")
    dataset = MultiFileDatasetUpgrade(data_dir, max_len=300)
    
    print(f"\n2. Checking name tracking...")
    # Check first few names
    for i in range(min(5, len(dataset))):
        name = dataset.get_name(i)
        print(f"   Sample {i}: {name}")
    
    # Create a mock cluster file for testing
    print(f"\n3. Creating mock cluster file...")
    mock_clstr = "/tmp/test_cluster.clstr"
    
    # Get first 10 names from dataset
    names = [dataset.get_name(i) for i in range(min(20, len(dataset)))]
    
    # Create mock clusters (2-3 sequences per cluster)
    with open(mock_clstr, 'w') as f:
        cluster_id = 0
        i = 0
        while i < len(names):
            f.write(f">Cluster {cluster_id}\n")
            # Add 2-3 sequences to this cluster
            n_in_cluster = min(2 if i % 2 == 0 else 3, len(names) - i)
            for j in range(n_in_cluster):
                if j == 0:
                    f.write(f"{j}\t100nt, >{names[i+j]}... *\n")
                else:
                    f.write(f"{j}\t100nt, >{names[i+j]}... at 94.5%\n")
            i += n_in_cluster
            cluster_id += 1
    
    print(f"   Created {mock_clstr}")
    
    # Parse cluster file
    print(f"\n4. Parsing cluster file...")
    name_to_cluster = parse_cdhit_clstr(mock_clstr)
    
    # Create splits
    print(f"\n5. Creating cluster-based splits...")
    train_idx, val_idx, test_idx = create_cluster_splits(
        dataset,
        name_to_cluster,
        train_frac=0.6,
        val_frac=0.2,
        seed=42
    )
    
    # Verify no overlap
    print(f"\n6. Verifying split integrity...")
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    
    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("   ❌ ERROR: Overlapping indices found!")
        if overlap_train_val:
            print(f"      Train/Val overlap: {len(overlap_train_val)} samples")
        if overlap_train_test:
            print(f"      Train/Test overlap: {len(overlap_train_test)} samples")
        if overlap_val_test:
            print(f"      Val/Test overlap: {len(overlap_val_test)} samples")
        return False
    else:
        print("   ✅ No overlapping indices")
    
    # Verify clusters are not split
    print(f"\n7. Verifying clusters are not split across sets...")
    cluster_in_train = set()
    cluster_in_val = set()
    cluster_in_test = set()
    
    for idx in train_idx:
        name = dataset.get_name(idx)
        if name in name_to_cluster:
            cluster_in_train.add(name_to_cluster[name])
    
    for idx in val_idx:
        name = dataset.get_name(idx)
        if name in name_to_cluster:
            cluster_in_val.add(name_to_cluster[name])
    
    for idx in test_idx:
        name = dataset.get_name(idx)
        if name in name_to_cluster:
            cluster_in_test.add(name_to_cluster[name])
    
    cluster_overlap = (cluster_in_train & cluster_in_val) | \
                      (cluster_in_train & cluster_in_test) | \
                      (cluster_in_val & cluster_in_test)
    
    if cluster_overlap:
        print(f"   ❌ ERROR: Clusters split across sets!")
        print(f"      Problematic clusters: {cluster_overlap}")
        return False
    else:
        print("   ✅ All clusters remain intact within splits")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_cluster_split()
    sys.exit(0 if success else 1)

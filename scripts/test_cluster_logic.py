#!/usr/bin/env python3
"""
Lightweight test for cluster parsing logic without heavy dependencies.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_cluster_parser():
    """Test CD-HIT cluster file parsing."""
    print("=" * 60)
    print("Testing CD-HIT Cluster File Parser")
    print("=" * 60)
    
    # Create a test cluster file
    test_clstr = "/tmp/test_parser.clstr"
    with open(test_clstr, 'w') as f:
        f.write(">Cluster 0\n")
        f.write("0\t119nt, >bpRNA_CRW_15639... *\n")
        f.write("1\t125nt, >bpRNA_CRW_15847... at 94.5%\n")
        f.write(">Cluster 1\n")
        f.write("0\t123nt, >bpRNA_CRW_15871... *\n")
        f.write(">Cluster 2\n")
        f.write("0\t120nt, >bpRNA_CRW_15994... *\n")
        f.write("1\t122nt, >bpRNA_CRW_16021... at 93.2%\n")
    
    print(f"\n1. Created test cluster file: {test_clstr}")
    
    # Parse it
    from src.cluster_split import parse_cdhit_clstr
    
    print(f"\n2. Parsing cluster file...")
    name_to_cluster = parse_cdhit_clstr(test_clstr)
    
    # Verify results
    print(f"\n3. Verifying results...")
    expected = {
        'bpRNA_CRW_15639': 0,
        'bpRNA_CRW_15847': 0,
        'bpRNA_CRW_15871': 1,
        'bpRNA_CRW_15994': 2,
        'bpRNA_CRW_16021': 2,
    }
    
    all_passed = True
    for name, expected_cluster in expected.items():
        actual_cluster = name_to_cluster.get(name)
        if actual_cluster == expected_cluster:
            print(f"   ✅ {name} -> cluster {actual_cluster}")
        else:
            print(f"   ❌ {name} -> cluster {actual_cluster} (expected {expected_cluster})")
            all_passed = False
    
    # Check cluster distribution
    print(f"\n4. Checking cluster distribution...")
    cluster_counts = {}
    for cluster_id in name_to_cluster.values():
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
    
    expected_counts = {0: 2, 1: 1, 2: 2}
    for cluster_id, expected_count in expected_counts.items():
        actual_count = cluster_counts.get(cluster_id, 0)
        if actual_count == expected_count:
            print(f"   ✅ Cluster {cluster_id}: {actual_count} sequences")
        else:
            print(f"   ❌ Cluster {cluster_id}: {actual_count} sequences (expected {expected_count})")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL PARSER TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
    
    return all_passed


def test_split_logic():
    """Test split logic without dataset dependency."""
    print("\n" + "=" * 60)
    print("Testing Split Logic")
    print("=" * 60)
    
    # Mock dataset class
    class MockDataset:
        def __init__(self, names):
            self.names = names
        
        def __len__(self):
            return len(self.names)
        
        def get_name(self, idx):
            return self.names[idx]
    
    # Create mock data with 10 samples in 4 clusters
    mock_names = [
        'seq_A1', 'seq_A2',        # cluster 0
        'seq_B1',                   # cluster 1
        'seq_C1', 'seq_C2', 'seq_C3',  # cluster 2
        'seq_D1', 'seq_D2',        # cluster 3
        'seq_E1', 'seq_E2',        # not in cluster (should be distributed)
    ]
    
    mock_dataset = MockDataset(mock_names)
    
    name_to_cluster = {
        'seq_A1': 0, 'seq_A2': 0,
        'seq_B1': 1,
        'seq_C1': 2, 'seq_C2': 2, 'seq_C3': 2,
        'seq_D1': 3, 'seq_D2': 3,
        # seq_E1 and seq_E2 not in clusters
    }
    
    print(f"\n1. Mock dataset: {len(mock_dataset)} samples")
    print(f"   Clusters: 4 (sizes: 2, 1, 3, 2)")
    print(f"   Unknown: 2 samples")
    
    from src.cluster_split import create_cluster_splits
    
    print(f"\n2. Creating splits (60% train, 20% val, 20% test)...")
    train_idx, val_idx, test_idx = create_cluster_splits(
        mock_dataset,
        name_to_cluster,
        train_frac=0.6,
        val_frac=0.2,
        seed=42
    )
    
    print(f"\n3. Verifying no index overlap...")
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    
    overlap = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
    if not overlap:
        print(f"   ✅ No overlapping indices")
    else:
        print(f"   ❌ Found overlapping indices: {overlap}")
        return False
    
    print(f"\n4. Verifying all indices present...")
    all_indices = set(range(len(mock_dataset)))
    covered = train_set | val_set | test_set
    if covered == all_indices:
        print(f"   ✅ All {len(mock_dataset)} samples covered")
    else:
        missing = all_indices - covered
        extra = covered - all_indices
        if missing:
            print(f"   ❌ Missing indices: {missing}")
        if extra:
            print(f"   ❌ Extra indices: {extra}")
        return False
    
    print(f"\n5. Verifying cluster integrity...")
    # Check that each cluster's samples all go to the same split
    cluster_splits = {}
    for idx in train_idx:
        name = mock_dataset.get_name(idx)
        if name in name_to_cluster:
            cluster_id = name_to_cluster[name]
            cluster_splits.setdefault(cluster_id, set()).add('train')
    
    for idx in val_idx:
        name = mock_dataset.get_name(idx)
        if name in name_to_cluster:
            cluster_id = name_to_cluster[name]
            cluster_splits.setdefault(cluster_id, set()).add('val')
    
    for idx in test_idx:
        name = mock_dataset.get_name(idx)
        if name in name_to_cluster:
            cluster_id = name_to_cluster[name]
            cluster_splits.setdefault(cluster_id, set()).add('test')
    
    all_intact = True
    for cluster_id, splits in cluster_splits.items():
        if len(splits) == 1:
            split_name = list(splits)[0]
            print(f"   ✅ Cluster {cluster_id} -> {split_name}")
        else:
            print(f"   ❌ Cluster {cluster_id} split across: {splits}")
            all_intact = False
    
    print("\n" + "=" * 60)
    if all_intact:
        print("✅ ALL SPLIT LOGIC TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
    
    return all_intact


if __name__ == "__main__":
    test1 = test_cluster_parser()
    test2 = test_split_logic()
    
    print("\n" + "=" * 60)
    if test1 and test2:
        print("✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)

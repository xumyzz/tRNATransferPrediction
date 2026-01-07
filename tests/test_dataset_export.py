#!/usr/bin/env python3
"""
Simple tests for dataset parsing and export functionality.
These tests don't require torch/numpy.
"""

import os
import sys
import tempfile
import json

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_st_parsing():
    """Test parsing of .st format"""
    print("Test 1: .st format parsing")
    
    # Create a temporary .st file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.st', delete=False) as f:
        f.write("#Name: test_seq1\n")
        f.write("#Length: 10\n")
        f.write("ACGUACGUAC\n")
        f.write("(((....)))\n")
        f.write("#Name: test_seq2\n")
        f.write("#Length: 8\n")
        f.write("GGGGCCCC\n")
        f.write("(((())))\n")
        st_file = f.name
    
    try:
        # Import after file creation
        from scripts.export_bprna_fasta import parse_st_file
        
        entries = list(parse_st_file(st_file))
        assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"
        assert entries[0][0] == "test_seq1", f"Expected 'test_seq1', got {entries[0][0]}"
        assert entries[1][0] == "test_seq2", f"Expected 'test_seq2', got {entries[1][0]}"
        
        print("  ✓ Parsed 2 entries correctly")
        print(f"  ✓ Entry names: {entries[0][0]}, {entries[1][0]}")
        
    finally:
        os.unlink(st_file)


def test_dbn_parsing():
    """Test parsing of .dbn format"""
    print("\nTest 2: .dbn format parsing")
    
    # Create a temporary .dbn file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dbn', delete=False) as f:
        f.write(">seq_a\n")
        f.write("ACGUACGU\n")
        f.write("((.....))\n")
        f.write("\n")
        f.write(">seq_b\n")
        f.write("GGGGCCCC\n")
        f.write("(((())))\n")
        dbn_file = f.name
    
    try:
        from scripts.export_bprna_fasta import parse_dbn_file
        
        entries = list(parse_dbn_file(dbn_file))
        assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"
        assert entries[0][0] == "seq_a", f"Expected 'seq_a', got {entries[0][0]}"
        assert entries[1][0] == "seq_b", f"Expected 'seq_b', got {entries[1][0]}"
        
        print("  ✓ Parsed 2 entries correctly")
        print(f"  ✓ Entry names: {entries[0][0]}, {entries[1][0]}")
        
    finally:
        os.unlink(dbn_file)


def test_validation():
    """Test validation logic"""
    print("\nTest 3: Validation logic")
    
    from scripts.export_bprna_fasta import is_valid_entry
    
    stats = {
        "total": 0,
        "kept": 0,
        "too_long": 0,
        "length_mismatch": 0,
        "too_many_n": 0,
        "invalid_bases": 0
    }
    
    # Valid entry
    valid, seq, struct = is_valid_entry(
        "test1", "ACGU", "(..)", max_len=10, n_threshold=0.2, stats=stats
    )
    assert valid, "Should be valid"
    assert seq == "ACGU", "Sequence should be normalized"
    print("  ✓ Valid entry accepted")
    
    # Too long
    stats_long = stats.copy()
    stats_long["total"] = 0
    valid, _, _ = is_valid_entry(
        "test2", "ACGU"*100, "."*400, max_len=10, n_threshold=0.2, stats=stats_long
    )
    assert not valid, "Should reject too long"
    assert stats_long["too_long"] == 1, "Should count as too_long"
    print("  ✓ Too long sequence rejected")
    
    # Length mismatch
    stats_mismatch = {"total": 0, "kept": 0, "too_long": 0, "length_mismatch": 0, 
                      "too_many_n": 0, "invalid_bases": 0}
    valid, _, _ = is_valid_entry(
        "test3", "ACGU", "...", max_len=10, n_threshold=0.2, stats=stats_mismatch
    )
    assert not valid, "Should reject length mismatch"
    assert stats_mismatch["length_mismatch"] == 1, "Should count as length_mismatch"
    print("  ✓ Length mismatch rejected")
    
    # Too many Ns
    stats_n = {"total": 0, "kept": 0, "too_long": 0, "length_mismatch": 0,
               "too_many_n": 0, "invalid_bases": 0}
    valid, _, _ = is_valid_entry(
        "test4", "NNNN", "....", max_len=10, n_threshold=0.2, stats=stats_n
    )
    assert not valid, "Should reject too many Ns"
    assert stats_n["too_many_n"] == 1, "Should count as too_many_n"
    print("  ✓ Too many Ns rejected")
    
    # Invalid bases
    stats_invalid = {"total": 0, "kept": 0, "too_long": 0, "length_mismatch": 0,
                     "too_many_n": 0, "invalid_bases": 0}
    valid, _, _ = is_valid_entry(
        "test5", "ACGX", "....", max_len=10, n_threshold=0.2, stats=stats_invalid
    )
    assert not valid, "Should reject invalid bases"
    assert stats_invalid["invalid_bases"] == 1, "Should count as invalid_bases"
    print("  ✓ Invalid bases rejected")
    
    # T->U normalization
    stats_norm = {"total": 0, "kept": 0, "too_long": 0, "length_mismatch": 0,
                  "too_many_n": 0, "invalid_bases": 0}
    valid, seq, _ = is_valid_entry(
        "test6", "ACGT", "....", max_len=10, n_threshold=0.2, stats=stats_norm
    )
    assert valid, "Should accept with T"
    assert seq == "ACGU", f"T should be converted to U, got {seq}"
    print("  ✓ T->U normalization works")


def test_cluster_parsing():
    """Test cluster file parsing"""
    print("\nTest 4: Cluster file parsing")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.clstr', delete=False) as f:
        f.write(">Cluster 0\n")
        f.write("0\t100aa, >seq1... *\n")
        f.write("1\t98aa, >seq2... at 95%\n")
        f.write(">Cluster 1\n")
        f.write("0\t80aa, >seq3... *\n")
        f.write(">Cluster 2\n")
        f.write("0\t120aa, >seq4... *\n")
        f.write("1\t119aa, >seq5... at 98%\n")
        clstr_file = f.name
    
    try:
        from scripts.cluster_utils import parse_cd_hit_clusters
        
        clusters = parse_cd_hit_clusters(clstr_file)
        assert len(clusters) == 3, f"Expected 3 clusters, got {len(clusters)}"
        assert len(clusters[0]) == 2, f"Cluster 0 should have 2 seqs, got {len(clusters[0])}"
        assert len(clusters[1]) == 1, f"Cluster 1 should have 1 seq, got {len(clusters[1])}"
        assert len(clusters[2]) == 2, f"Cluster 2 should have 2 seqs, got {len(clusters[2])}"
        
        print(f"  ✓ Parsed 3 clusters correctly")
        print(f"  ✓ Cluster sizes: {[len(c) for c in clusters]}")
        print(f"  ✓ Cluster 0: {clusters[0]}")
        
    finally:
        os.unlink(clstr_file)


def test_export_integration():
    """Test full export pipeline"""
    print("\nTest 5: Full export pipeline")
    
    # Create test directory with mixed .st and .dbn files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create .st file
        st_file = os.path.join(tmpdir, "test.st")
        with open(st_file, 'w') as f:
            f.write("#Name: st_seq1\n")
            f.write("#Length: 8\n")
            f.write("ACGUACGU\n")
            f.write("((....))\n")
        
        # Create .dbn file
        dbn_file = os.path.join(tmpdir, "test.dbn")
        with open(dbn_file, 'w') as f:
            f.write(">dbn_seq1\n")
            f.write("GGGGCCCC\n")
            f.write("(((())))\n")
        
        # Run export
        import subprocess
        
        out_fasta = os.path.join(tmpdir, "out.fasta")
        out_stats = os.path.join(tmpdir, "stats.json")
        
        cmd = [
            sys.executable,
            "scripts/export_bprna_fasta.py",
            "--input", tmpdir,
            "--max_len", "600",
            "--out_fasta", out_fasta,
            "--stats_out", out_stats
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        
        # Check output
        assert os.path.exists(out_fasta), "FASTA file should be created"
        assert os.path.exists(out_stats), "Stats file should be created"
        
        with open(out_fasta) as f:
            fasta_content = f.read()
            assert ">st_seq1" in fasta_content, "Should contain st_seq1"
            assert ">dbn_seq1" in fasta_content, "Should contain dbn_seq1"
        
        with open(out_stats) as f:
            stats = json.load(f)
            assert stats["kept"] == 2, f"Should keep 2 sequences, got {stats['kept']}"
        
        print("  ✓ Export created FASTA file")
        print("  ✓ Export created stats file")
        print(f"  ✓ Exported {stats['kept']} sequences")


def run_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running dataset and export tests")
    print("=" * 60)
    
    try:
        test_st_parsing()
        test_dbn_parsing()
        test_validation()
        test_cluster_parsing()
        test_export_integration()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(run_tests())

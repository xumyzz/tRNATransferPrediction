#!/usr/bin/env python3
"""
Test edge cases and robustness of parsing functions.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.cluster_utils import parse_cd_hit_clusters


def test_cluster_parsing_with_periods():
    """Test cluster parsing with sequence names containing periods"""
    print("Test: Cluster parsing with periods in names")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.clstr', delete=False) as f:
        f.write(">Cluster 0\n")
        f.write("0\t100aa, >seq1.v2.final... *\n")
        f.write("1\t98aa, >seq2.test... at 95%\n")
        f.write(">Cluster 1\n")
        f.write("0\t80aa, >my.complex.seq.name... *\n")
        clstr_file = f.name
    
    try:
        clusters = parse_cd_hit_clusters(clstr_file)
        assert len(clusters) == 2, f"Expected 2 clusters, got {len(clusters)}"
        assert clusters[0][0] == "seq1.v2.final", f"Expected 'seq1.v2.final', got '{clusters[0][0]}'"
        assert clusters[0][1] == "seq2.test", f"Expected 'seq2.test', got '{clusters[0][1]}'"
        assert clusters[1][0] == "my.complex.seq.name", f"Expected 'my.complex.seq.name', got '{clusters[1][0]}'"
        
        print("  ✓ Correctly parsed names with periods")
        print(f"  ✓ Names: {clusters[0]}, {clusters[1]}")
        
    finally:
        os.unlink(clstr_file)


def test_dbn_default_names():
    """Test .dbn parsing with missing names uses unique defaults"""
    print("\nTest: .dbn default name generation")
    
    from scripts.export_bprna_fasta import parse_dbn_file
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dbn', delete=False, 
                                     prefix='testfile') as f:
        f.write(">\n")  # Empty name
        f.write("ACGU\n")
        f.write("....\n")
        f.write("\n")
        f.write(">\n")  # Another empty name
        f.write("GGCC\n")
        f.write("(())\n")
        dbn_file = f.name
        basename = os.path.splitext(os.path.basename(f.name))[0]
    
    try:
        entries = list(parse_dbn_file(dbn_file))
        assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"
        
        # Check that default names are unique and contain basename
        name1, name2 = entries[0][0], entries[1][0]
        assert basename in name1, f"Basename '{basename}' not in '{name1}'"
        assert basename in name2, f"Basename '{basename}' not in '{name2}'"
        assert name1 != name2, f"Names should be unique: '{name1}' == '{name2}'"
        
        print(f"  ✓ Generated unique default names")
        print(f"  ✓ Name 1: {name1}")
        print(f"  ✓ Name 2: {name2}")
        
    finally:
        os.unlink(dbn_file)


def run_edge_case_tests():
    """Run all edge case tests"""
    print("=" * 60)
    print("Running edge case tests")
    print("=" * 60)
    
    try:
        test_cluster_parsing_with_periods()
        test_dbn_default_names()
        
        print("\n" + "=" * 60)
        print("All edge case tests passed! ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(run_edge_case_tests())

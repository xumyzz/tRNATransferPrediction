#!/usr/bin/env python3
"""
Tests for content-based format sniffing and pseudoknot filtering.
"""

import os
import sys
import tempfile
import json
import subprocess

# Add parent and scripts to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

from scripts.format_utils import sniff_format, has_pseudoknot


def test_sniff_st_format():
    """Test sniffing .st format files"""
    print("Test 1: Sniff .st format")
    
    # Create a file with .st content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dbn', delete=False) as f:
        f.write("#Name: test_seq1\n")
        f.write("#Length: 10\n")
        f.write("ACGUACGUAC\n")
        f.write("(((....)))\n")
        test_file = f.name
    
    try:
        result = sniff_format(test_file)
        assert result == "st", f"Expected 'st', got '{result}'"
        print("  ✓ Correctly identified .st format in .dbn file")
    finally:
        os.unlink(test_file)


def test_sniff_dbn_format():
    """Test sniffing .dbn format files"""
    print("\nTest 2: Sniff .dbn format")
    
    # Create a file with .dbn content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.st', delete=False) as f:
        f.write(">seq_a\n")
        f.write("ACGUACGU\n")
        f.write("((.....))\n")
        test_file = f.name
    
    try:
        result = sniff_format(test_file)
        assert result == "dbn", f"Expected 'dbn', got '{result}'"
        print("  ✓ Correctly identified .dbn format in .st file")
    finally:
        os.unlink(test_file)


def test_sniff_unknown_format():
    """Test sniffing unknown format files"""
    print("\nTest 3: Sniff unknown format")
    
    # Create a file with unknown content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is just some text\n")
        f.write("Not a valid format\n")
        test_file = f.name
    
    try:
        result = sniff_format(test_file)
        assert result is None, f"Expected None, got '{result}'"
        print("  ✓ Correctly returned None for unknown format")
    finally:
        os.unlink(test_file)


def test_has_pseudoknot():
    """Test pseudoknot detection"""
    print("\nTest 4: Pseudoknot detection")
    
    # Test various structures
    cases = [
        ("(((...)))", False, "simple structure"),
        ("(((...[[[)))]]]", True, "structure with []"),
        ("(((...))){{{...}}}", True, "structure with {}"),
        ("(((...)))<<<<>>>>", True, "structure with <>"),
        ("........", False, "no pairs"),
        ("((((....))))", False, "nested pairs only"),
        ("((..[[..))..]]", True, "mixed [ and ("),
    ]
    
    for struct, expected, desc in cases:
        result = has_pseudoknot(struct)
        assert result == expected, f"Failed for {desc}: expected {expected}, got {result}"
        print(f"  ✓ {desc}: {result}")


def test_export_with_dbn_containing_st():
    """Test export script with .dbn file containing .st format"""
    print("\nTest 5: Export with mis-labeled .dbn file")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a .dbn file with .st content
        dbn_file = os.path.join(tmpdir, "test.dbn")
        with open(dbn_file, 'w') as f:
            f.write("#Name: st_in_dbn_1\n")
            f.write("#Length: 8\n")
            f.write("ACGUACGU\n")
            f.write("((....))\n")
            f.write("#Name: st_in_dbn_2\n")
            f.write("#Length: 10\n")
            f.write("GGGGGGCCCC\n")
            f.write("((((())))) \n")  # Note: extra space to test trimming
        
        # Run export
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
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        
        # Check output
        assert os.path.exists(out_fasta), "FASTA file should be created"
        assert os.path.exists(out_stats), "Stats file should be created"
        
        with open(out_fasta) as f:
            fasta_content = f.read()
            assert ">st_in_dbn_1" in fasta_content, "Should contain st_in_dbn_1"
            assert ">st_in_dbn_2" in fasta_content, "Should contain st_in_dbn_2"
        
        with open(out_stats) as f:
            stats = json.load(f)
            assert stats["kept"] == 2, f"Should keep 2 sequences, got {stats['kept']}"
            assert stats["sniffed_st_in_dbn"] == 1, f"Should detect 1 .dbn file as .st, got {stats['sniffed_st_in_dbn']}"
        
        print("  ✓ Export correctly parsed .dbn file with .st format")
        print(f"  ✓ Exported {stats['kept']} sequences")
        print(f"  ✓ Detected {stats['sniffed_st_in_dbn']} mis-labeled file")


def test_pseudoknot_filtering():
    """Test pseudoknot filtering in export"""
    print("\nTest 6: Pseudoknot filtering")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file with pseudoknot structures
        st_file = os.path.join(tmpdir, "test.st")
        with open(st_file, 'w') as f:
            # Entry without pseudoknot
            f.write("#Name: no_pseudoknot\n")
            f.write("ACGUACGU\n")
            f.write("((....))\n")
            # Entry with pseudoknot (using [ and ] brackets)
            f.write("#Name: has_pseudoknot\n")
            f.write("ACGUACGUAC\n")
            f.write("((..[[.)]]\n")
            # Another without pseudoknot
            f.write("#Name: no_pseudoknot_2\n")
            f.write("GGGGCCCC\n")
            f.write("(((())))\n")
        
        # Test with filtering (default)
        out_fasta_filtered = os.path.join(tmpdir, "out_filtered.fasta")
        out_stats_filtered = os.path.join(tmpdir, "stats_filtered.json")
        
        cmd_filtered = [
            sys.executable,
            "scripts/export_bprna_fasta.py",
            "--input", tmpdir,
            "--max_len", "600",
            "--out_fasta", out_fasta_filtered,
            "--stats_out", out_stats_filtered
        ]
        
        subprocess.run(
            cmd_filtered, 
            capture_output=True, 
            text=True, 
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        
        with open(out_stats_filtered) as f:
            stats_filtered = json.load(f)
            assert stats_filtered["kept"] == 2, f"Should keep 2 non-pseudoknot sequences, got {stats_filtered['kept']}"
            assert stats_filtered["pseudoknot_filtered"] == 1, f"Should filter 1 pseudoknot, got {stats_filtered['pseudoknot_filtered']}"
        
        print("  ✓ Filtered pseudoknot structures (default)")
        print(f"  ✓ Kept {stats_filtered['kept']} sequences")
        print(f"  ✓ Filtered {stats_filtered['pseudoknot_filtered']} pseudoknot structures")
        
        # Test with --allow_pseudoknot
        out_fasta_allowed = os.path.join(tmpdir, "out_allowed.fasta")
        out_stats_allowed = os.path.join(tmpdir, "stats_allowed.json")
        
        cmd_allowed = [
            sys.executable,
            "scripts/export_bprna_fasta.py",
            "--input", tmpdir,
            "--max_len", "600",
            "--allow_pseudoknot",
            "--out_fasta", out_fasta_allowed,
            "--stats_out", out_stats_allowed
        ]
        
        subprocess.run(
            cmd_allowed, 
            capture_output=True, 
            text=True, 
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        
        with open(out_stats_allowed) as f:
            stats_allowed = json.load(f)
            assert stats_allowed["kept"] == 3, f"Should keep all 3 sequences, got {stats_allowed['kept']}"
            assert stats_allowed["pseudoknot_filtered"] == 0, f"Should not filter pseudoknots, got {stats_allowed['pseudoknot_filtered']}"
        
        print("  ✓ Allowed pseudoknot structures with --allow_pseudoknot")
        print(f"  ✓ Kept all {stats_allowed['kept']} sequences")


def run_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running format sniffing and pseudoknot tests")
    print("=" * 60)
    
    try:
        test_sniff_st_format()
        test_sniff_dbn_format()
        test_sniff_unknown_format()
        test_has_pseudoknot()
        test_export_with_dbn_containing_st()
        test_pseudoknot_filtering()
        
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

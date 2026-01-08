"""
Utility functions for RNA structure file format detection and validation.
This module has no heavy dependencies (no torch, numpy).
"""


def sniff_format(fpath, max_lines=50):
    """
    Detect the format of an RNA structure file by examining its content.
    
    This function reads the first N non-empty lines of a file and determines
    whether it's in bpRNA .st format or FASTA-like .dbn format.
    
    Args:
        fpath: Path to the file to sniff
        max_lines: Maximum number of non-empty lines to examine (default: 50)
    
    Returns:
        str: "st" if file appears to be bpRNA .st format (lines start with #Name:)
             "dbn" if file appears to be FASTA-like .dbn format (lines start with >)
             None if format cannot be determined
    
    Examples:
        >>> sniff_format("data.st")
        'st'
        >>> sniff_format("data.dbn")  # Actually contains st-format
        'st'
    """
    try:
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            lines_checked = 0
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                
                # Check for st format markers
                if stripped.startswith("#Name:"):
                    return "st"
                
                # Check for dbn format markers (FASTA-like)
                if stripped.startswith(">"):
                    return "dbn"
                
                lines_checked += 1
                if lines_checked >= max_lines:
                    break
        
        # No recognizable format found
        return None
        
    except Exception:
        # If we can't read the file, return None
        return None


def has_pseudoknot(structure):
    """
    Check if a structure string contains pseudoknot notation.
    
    Pseudoknots are represented using brackets beyond standard parentheses:
    [], {}, <> characters indicate pseudoknot base pairs.
    
    Args:
        structure: Dot-bracket structure string
    
    Returns:
        bool: True if structure contains pseudoknot characters, False otherwise
    
    Examples:
        >>> has_pseudoknot("(((...)))")
        False
        >>> has_pseudoknot("(((...[[[)))]]]")
        True
    """
    pseudoknot_chars = set('[]{}<>')
    return any(c in pseudoknot_chars for c in structure)

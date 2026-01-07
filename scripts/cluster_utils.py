"""
Utility functions for cluster-based splitting and data processing.
This module has no heavy dependencies (no torch, numpy).
"""


def parse_cd_hit_clusters(clstr_path):
    """
    Parse cd-hit-est .clstr output file.
    
    Args:
        clstr_path: Path to .clstr file
    
    Returns:
        List of clusters, where each cluster is a list of sequence names.
    
    Example .clstr format:
        >Cluster 0
        0	100aa, >seq1... *
        1	98aa, >seq2... at 95%
        >Cluster 1
        0	80aa, >seq3... *
    """
    clusters = []
    current_cluster = []
    
    with open(clstr_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>Cluster'):
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = []
            else:
                # Parse sequence name from cluster entry
                # Format: "0	123nt, >seq_name... at 100%"
                if '>' in line:
                    name_part = line.split('>')[1]
                    # Remove trailing "..." and anything after it
                    # Use rsplit to handle names with periods like "seq1.v2.final"
                    if '...' in name_part:
                        name = name_part.split('...')[0].strip()
                    else:
                        # Fallback: split on whitespace and take first part
                        name = name_part.split()[0].strip()
                    current_cluster.append(name)
        
        # Add last cluster
        if current_cluster:
            clusters.append(current_cluster)
    
    return clusters

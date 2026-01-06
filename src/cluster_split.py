"""
Utilities for cluster-based data splitting to prevent data leakage.
"""

import json
import random
from typing import Dict, List, Tuple


def parse_cdhit_clstr(clstr_path: str) -> Dict[str, int]:
    """
    Parse a CD-HIT .clstr file to create a name -> cluster_id mapping.
    
    CD-HIT .clstr format:
        >Cluster 0
        0    123nt, >seq1... *
        1    125nt, >seq2... at 95.2%
        >Cluster 1
        ...
    
    Args:
        clstr_path: Path to .clstr file
    
    Returns:
        Dict mapping sequence name to cluster ID
    """
    name_to_cluster = {}
    current_cluster_id = -1
    
    with open(clstr_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # New cluster starts
            if line.startswith(">Cluster"):
                try:
                    current_cluster_id = int(line.split()[-1])
                except (ValueError, IndexError):
                    # Try alternate parsing
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "Cluster" and i + 1 < len(parts):
                            try:
                                current_cluster_id = int(parts[i + 1])
                                break
                            except ValueError:
                                pass
                continue
            
            # Member line format: "0    123nt, >seq_name... *" or "1    125nt, >seq_name... at 95.2%"
            if line and current_cluster_id >= 0:
                # Extract sequence name from between '>' and '...'
                if '>' in line:
                    # Find the '>' and extract until first whitespace or '...'
                    start_idx = line.index('>')
                    name_part = line[start_idx + 1:]
                    
                    # Extract name up to first whitespace or '...'
                    if '...' in name_part:
                        name = name_part.split('...')[0]
                    else:
                        name = name_part.split()[0]
                    
                    # Remove trailing punctuation if any
                    name = name.rstrip('.,;:')
                    
                    name_to_cluster[name] = current_cluster_id
    
    print(f"📊 解析 {clstr_path}:")
    print(f"  - 找到 {len(name_to_cluster)} 个序列")
    print(f"  - 分布在 {len(set(name_to_cluster.values()))} 个簇中")
    
    return name_to_cluster


def create_cluster_splits(
    dataset,
    name_to_cluster: Dict[str, int],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create train/val/test splits based on cluster membership.
    
    Ensures that all samples from the same cluster go to the same split,
    preventing data leakage.
    
    Args:
        dataset: Dataset object with get_name(idx) method
        name_to_cluster: Dict mapping sample name to cluster ID
        train_frac: Fraction for training (default 0.8)
        val_frac: Fraction for validation (default 0.1)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    random.seed(seed)
    
    # Group indices by cluster
    cluster_to_indices = {}
    unknown_indices = []
    
    for idx in range(len(dataset)):
        name = dataset.get_name(idx)
        cluster_id = name_to_cluster.get(name)
        
        if cluster_id is not None:
            if cluster_id not in cluster_to_indices:
                cluster_to_indices[cluster_id] = []
            cluster_to_indices[cluster_id].append(idx)
        else:
            unknown_indices.append(idx)
    
    # Get all cluster IDs and shuffle
    cluster_ids = list(cluster_to_indices.keys())
    random.shuffle(cluster_ids)
    
    # Calculate split points
    n_clusters = len(cluster_ids)
    n_train = int(n_clusters * train_frac)
    n_val = int(n_clusters * val_frac)
    
    # Split clusters
    train_clusters = cluster_ids[:n_train]
    val_clusters = cluster_ids[n_train:n_train + n_val]
    test_clusters = cluster_ids[n_train + n_val:]
    
    # Collect indices for each split
    train_indices = []
    val_indices = []
    test_indices = []
    
    for cluster_id in train_clusters:
        train_indices.extend(cluster_to_indices[cluster_id])
    
    for cluster_id in val_clusters:
        val_indices.extend(cluster_to_indices[cluster_id])
    
    for cluster_id in test_clusters:
        test_indices.extend(cluster_to_indices[cluster_id])
    
    # Handle unknown samples - distribute proportionally
    if unknown_indices:
        random.shuffle(unknown_indices)
        n_unknown = len(unknown_indices)
        n_unknown_train = int(n_unknown * train_frac)
        n_unknown_val = int(n_unknown * val_frac)
        
        train_indices.extend(unknown_indices[:n_unknown_train])
        val_indices.extend(unknown_indices[n_unknown_train:n_unknown_train + n_unknown_val])
        test_indices.extend(unknown_indices[n_unknown_train + n_unknown_val:])
        
        print(f"⚠️  {len(unknown_indices)} 个样本未在聚类文件中找到，按比例分配")
    
    # Print statistics
    print("\n" + "=" * 50)
    print("📊 聚类分割统计:")
    print(f"  训练集: {len(train_indices)} 样本 ({len(train_clusters)} 个簇)")
    print(f"  验证集: {len(val_indices)} 样本 ({len(val_clusters)} 个簇)")
    print(f"  测试集: {len(test_indices)} 样本 ({len(test_clusters)} 个簇)")
    print(f"  总计:   {len(train_indices) + len(val_indices) + len(test_indices)} 样本")
    print("=" * 50 + "\n")
    
    return train_indices, val_indices, test_indices


def save_split_config(
    save_path: str,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    metadata: dict = None
):
    """
    Save split configuration to JSON for reproducibility.
    
    Args:
        save_path: Path to save JSON file
        train_indices: List of training indices
        val_indices: List of validation indices
        test_indices: List of test indices
        metadata: Optional dict with additional metadata
    """
    config = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "metadata": metadata or {}
    }
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ 保存分割配置到 {save_path}")


def load_split_config(load_path: str) -> Tuple[List[int], List[int], List[int], dict]:
    """
    Load split configuration from JSON.
    
    Args:
        load_path: Path to JSON file
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices, metadata)
    """
    with open(load_path, 'r') as f:
        config = json.load(f)
    
    return (
        config["train_indices"],
        config["val_indices"],
        config["test_indices"],
        config.get("metadata", {})
    )

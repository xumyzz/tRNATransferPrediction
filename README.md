# tRNA Transfer Prediction

Deep learning model for RNA secondary structure prediction using ResNet and Transformer architecture.

## Overview

This repository implements RNA secondary structure prediction with support for:
- **TR0 dataset**: Standard training on curated tRNA dataset
- **bpRNA-1m pretraining**: Large-scale pretraining on bpRNA-1m dataset with cluster-disjoint splits

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- CD-HIT (for clustering, optional)

## Quick Start

### Training on TR0 (Standard Workflow)

```bash
# Using the default config
python main.py
```

The default configuration is in `src/config.py`. Modify the paths and hyperparameters as needed.

### Training on bpRNA-1m (Pretraining Workflow)

The bpRNA-1m pretraining workflow uses cluster-based splitting to ensure that similar sequences are not split across train/val/test sets, preventing data leakage.

**Important Notes:**
- **Content-based format detection**: Files are parsed based on their content, not their extension. Some bpRNA downloads may have `.dbn` extension but contain `#Name:` records (bpRNA .st format). The system automatically detects and parses them correctly.
- **Pseudoknot filtering**: By default, structures containing pseudoknot notation (`[]`, `{}`, `<>`) are filtered out. Use `--allow_pseudoknot` to keep them.

#### Step 1: Export FASTA for Clustering

Export sequences from your bpRNA dataset to FASTA format:

```bash
python scripts/export_bprna_fasta.py \
    --input data/bpRNA_CRW \
    --input data/bpRNA_RFAM \
    --input data/bpRNA_tmRNA \
    --max_len 600 \
    --out_fasta bprna_1m.fasta \
    --out_names bprna_1m_names.txt \
    --stats_out bprna_1m_stats.json
```

**Options:**
- `--input`: Can be specified multiple times; accepts files or directories
- `--max_len`: Maximum sequence length (default: 600)
- `--allow_pseudoknot`: Allow structures with pseudoknot notation (default: filter them out)
- `--out_fasta`: Output FASTA file (required)
- `--out_names`: Optional text file with sequence names (one per line)
- `--stats_out`: Optional JSON file with filtering statistics

The script automatically discovers `.st` and `.dbn` files in directories and detects their format by content.

#### Step 2: Cluster Sequences with CD-HIT

Use CD-HIT to cluster sequences by similarity. This prevents similar sequences from appearing in both training and validation sets.

```bash
# Install CD-HIT if not already installed
# conda install -c bioconda cd-hit
# or download from: https://github.com/weizhongli/cdhit

# Run clustering at 90% sequence identity
cd-hit-est \
    -i bprna_1m.fasta \
    -o bprna_1m_clustered.fasta \
    -c 0.90 \
    -n 8 \
    -M 16000 \
    -T 8 \
    -d 0
```

**CD-HIT Parameters:**
- `-c 0.90`: 90% sequence identity threshold (adjust based on your needs)
- `-n 8`: Word size (8 for ~90% identity, 7 for ~88%, 5 for ~80%)
- `-M 16000`: Memory limit in MB
- `-T 8`: Number of threads
- `-d 0`: Length of description in output (0 = unlimited)

For sequences ≤600bp, the recommended word size is:
- 90% identity: `-c 0.90 -n 8`
- 85% identity: `-c 0.85 -n 7`
- 80% identity: `-c 0.80 -n 5`

This will produce `bprna_1m_clustered.fasta.clstr` which contains the cluster assignments.

#### Step 3: Train with Cluster-Based Split

Use the cluster file to create train/val splits that respect cluster boundaries:

```bash
python train_with_args.py \
    --data_dir data/bpRNA_1m \
    --max_len 600 \
    --clstr_path bprna_1m_clustered.fasta.clstr \
    --train_frac 0.8 \
    --val_frac 0.1 \
    --split_out splits.json \
    --batch_size 2 \
    --accum_steps 32 \
    --epochs 20 \
    --lr 0.0001 \
    --save_dir checkpoints_bprna1m
```

**Key Arguments:**
- `--data_dir`: Directory containing `.st` and `.dbn` files
- `--max_len`: Maximum sequence length (must match export step)
- `--clstr_path`: Path to CD-HIT `.clstr` file
- `--train_frac`: Fraction of **clusters** for training (default: 0.8)
- `--val_frac`: Fraction of **clusters** for validation (default: 0.1)
- `--split_out`: Save split indices to JSON (optional, for reproducibility)
- `--batch_size`: Batch size (reduce if OOM)
- `--accum_steps`: Gradient accumulation steps (effective batch = batch_size × accum_steps)

**Without cluster-based split** (random split):
```bash
python train_with_args.py \
    --data_dir data/TR0 \
    --max_len 300 \
    --train_frac 0.9 \
    --val_frac 0.1 \
    --epochs 10
```

## Data Formats

### .st Format (bpRNA Standard)

```
#Name: bpRNA_CRW_29003
#Length: 75
#PageNumber: 1
UGGCCCCAUCGACUAGCGGUUAGGUCACCGGCCUUUCAAGCCGGCGGCGGGGGUUCGAGUCCCCCUGGGGUCACC
.(((((((..((((........))))((((((.......))))))...(((((.......))))))))))))...
```

### .dbn Format (FASTA-like)

```
>seq_name_1
ACGUACGUACGU
(((...)))...

>seq_name_2
GGGGCCCC
(((()))).
```

**Format Detection**: Both formats are automatically detected and parsed by the dataset loader based on content, not file extension. Files with `.dbn` extension may actually contain `.st` format data (starting with `#Name:`), and they will be parsed correctly.

**Pseudoknot Notation**: Structures may contain pseudoknot brackets (`[]`, `{}`, `<>`) in addition to standard parentheses `()`. By default, entries with pseudoknots are filtered out during loading. To include them, use the `--allow_pseudoknot` flag (dataset) or set `allow_pseudoknot=True` (Python API). Note that the model may not handle pseudoknots correctly as they represent non-nested base pairs.

## Memory and Performance Optimization

RNA structure prediction has **O(L²)** memory complexity due to the contact matrix. For long sequences:

### Memory-Saving Strategies

1. **Reduce max_len**: 
   - L=300: ~4GB per sample
   - L=600: ~16GB per sample
   
2. **Reduce batch_size**: Use `batch_size=1` or `2` with high `accum_steps`

3. **Enable Mixed Precision (AMP)**:
   ```python
   # In train_with_args.py, add:
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   
   with autocast():
       logits = model(seqs, mask=masks)
       loss = compute_masked_loss(...)
   ```

4. **Use Gradient Checkpointing**: Trade computation for memory

5. **Multi-GPU Training**: Distribute batch across GPUs

### Example for 2× 24GB GPUs with L=600

```bash
python train_with_args.py \
    --data_dir data/bpRNA_1m \
    --max_len 600 \
    --batch_size 1 \
    --accum_steps 64 \
    --epochs 20
```

This gives an effective batch size of 64 while keeping memory per GPU manageable.

## File Structure

```
.
├── src/
│   ├── dataset.py          # Dataset classes with .st and .dbn support
│   ├── model.py            # Model architecture
│   ├── train.py            # Original training script
│   ├── config.py           # Configuration
│   └── utils.py            # Loss and metric functions
├── scripts/
│   └── export_bprna_fasta.py  # Export to FASTA for clustering
├── data/
│   ├── TR0/                # Training data
│   ├── TS0/                # Test data
│   └── VL0/                # Validation data
├── main.py                 # Entry point for standard training
├── train_with_args.py      # Enhanced training with cluster splits
└── README.md
```

## Dataset Classes

- `BpRNAProcessor`: Converts sequences and structures to tensors
- `MultiFileDataset`: Original dataset class
- `MultiFileDatasetUpgrade`: Enhanced dataset with:
  - Support for both `.st` and `.dbn` formats
  - Name tracking for each sample
  - Robust validation (length, N threshold, valid bases)
  - Sequence normalization (uppercase, T→U)

## Validation Checks

The dataset loader applies the following filters:
- **Length**: Sequences longer than `max_len` are rejected
- **Length match**: Sequence and structure must have equal length
- **Valid bases**: Only A, C, G, U, N are allowed
- **N threshold**: Sequences with >20% N bases are rejected (configurable)

## Cluster-Based Splitting

Cluster-based splitting ensures:
1. All sequences in a cluster stay together in the same split
2. No data leakage between train/val/test sets
3. Better generalization to unseen sequences

The split is performed at the **cluster level**, not the sequence level. This means:
- If 80% of clusters are used for training, the actual sequence fraction may differ slightly
- The split respects natural sequence families

## Citation

If you use this code, please cite the original bpRNA and tRNA datasets:

```
# Add relevant citations here
```

## License

[Specify license]

## Troubleshooting

### Out of Memory Errors

- Reduce `max_len`
- Reduce `batch_size` to 1
- Increase `accum_steps` to maintain effective batch size
- Enable mixed precision training (AMP)

### Cluster File Not Found

Make sure the `.clstr` file path is correct and was generated by cd-hit-est.

### Import Errors

Ensure all dependencies are installed:
```bash
pip install torch numpy
```

### Data Loading Issues

Check that:
- Data files have correct extensions (`.st` or `.dbn`)
- Files are not corrupted
- Paths in `--data_dir` or `--input` are correct

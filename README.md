# tRNA Transfer Prediction

This repository implements a deep learning model for tRNA secondary structure prediction with support for cluster-based data splitting to prevent data leakage during cross-family/generalization evaluation.

## Features

- **bpRNA format parsing**: Reads `.st` files with bpRNA-style records
- **Sequence length filtering**: Filters sequences up to MAX_LEN (default 300)
- **Cluster-based splitting**: Prevents data leakage by ensuring sequences from the same cluster stay in the same split
- **Flexible training**: Supports both random splitting (backward compatible) and cluster-based splitting

## Installation

```bash
# Clone the repository
git clone https://github.com/xumyzz/tRNATransferPrediction.git
cd tRNATransferPrediction

# Install dependencies (PyTorch, NumPy, etc.)
pip install torch numpy
```

## Quick Start

### 1. Random Split Training (Default)

```bash
# Train with random split (no data leakage prevention)
python main.py
```

### 2. Cluster-Based Training (Recommended for Evaluation)

To prevent data leakage when evaluating cross-family generalization:

#### Step 1: Export FASTA from TR0 data

```bash
python scripts/export_tr0_fasta.py \
  --data_dir data/TR0 \
  --max_len 300 \
  --out_fasta tr0_sequences.fasta \
  --out_names tr0_names.txt
```

This script:
- Parses all `.st` files in the data directory
- Applies the same validation and filtering as training
- Outputs sequences in FASTA format with headers matching `#Name` values
- Normalizes sequences (uppercase, T→U)

#### Step 2: Run CD-HIT clustering

```bash
cd-hit-est \
  -i tr0_sequences.fasta \
  -o tr0_cdhit95 \
  -c 0.95 \
  -n 10 \
  -d 0 \
  -M 0 \
  -T 0
```

Parameters:
- `-c 0.95`: Sequence identity threshold (95%)
- `-n 10`: Word length (10 for threshold ≥0.9)
- `-d 0`: Full description in .clstr file
- `-M 0`: Unlimited memory
- `-T 0`: Use all CPU threads

This produces `tr0_cdhit95.clstr` which maps sequences to cluster IDs.

#### Step 3: Train with cluster-based split

```bash
python main.py \
  --clstr_path tr0_cdhit95.clstr \
  --split_seed 42 \
  --train_frac 0.8 \
  --val_frac 0.1 \
  --split_out splits.json
```

This ensures:
- All sequences from the same cluster go to the same split (train/val/test)
- No data leakage between splits
- Reproducible splits via `--split_seed`
- Split configuration saved to `splits.json` for reproducibility

## Command-Line Arguments

### `main.py` (Training)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--clstr_path` | str | None | Path to CD-HIT .clstr file for cluster-based splitting |
| `--split_seed` | int | 42 | Random seed for reproducible splits |
| `--train_frac` | float | 0.8 | Fraction of clusters for training |
| `--val_frac` | float | 0.1 | Fraction of clusters for validation |
| `--split_out` | str | None | Path to save split configuration JSON |

### `scripts/export_tr0_fasta.py` (FASTA Export)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_dir` | str | **required** | Directory containing .st files (or single .st file) |
| `--max_len` | int | 300 | Maximum sequence length (same as training) |
| `--out_fasta` | str | **required** | Output FASTA file path |
| `--out_names` | str | None | Optional output file for list of names |

## Configuration

Edit `src/config.py` to customize training parameters:

```python
class Config:
    DATA_DIR = "/path/to/TR0"  # Your data directory
    MODEL_SAVE_DIR = "/path/to/checkpoints"
    MAX_LEN = 300              # Maximum sequence length
    BATCH_SIZE = 2
    EPOCHS = 10
    LR = 0.0001
    # ... other parameters
```

## Data Format

The parser expects bpRNA-format `.st` files with structure:

```
#Name: bpRNA_CRW_15639
#Length: 119
#PageNumber: 1
UCCCUGGUGAAAUUAGCGC...
((((((((((.....(((((...
```

Each file may contain multiple entries. The parser:
1. Extracts the `#Name:` value as the sample identifier
2. Validates sequence length (≤ MAX_LEN)
3. Validates sequence-structure length matching
4. Filters out sequences with >20% N content

## Preventing Data Leakage

**Why cluster-based splitting?**

Random splitting may place highly similar sequences in different splits (train/val/test), causing:
- **Data leakage**: The model sees similar sequences during training and testing
- **Overestimated performance**: High accuracy on test set doesn't reflect true generalization
- **Poor cross-family performance**: Model fails on truly novel sequences

**Solution: Cluster-based splitting**

By grouping similar sequences (via CD-HIT at 95% identity) and keeping clusters together in splits:
- ✅ No similar sequences across train/val/test
- ✅ True cross-family/generalization evaluation
- ✅ More realistic performance metrics

## File Structure

```
tRNATransferPrediction/
├── main.py                    # Training entry point
├── src/
│   ├── train.py              # Training logic with cluster split support
│   ├── dataset.py            # Dataset parser with name tracking
│   ├── cluster_split.py      # Cluster-based splitting utilities
│   ├── model.py              # Model architecture
│   ├── config.py             # Configuration
│   └── utils.py              # Utility functions
├── scripts/
│   └── export_tr0_fasta.py   # FASTA export for clustering
└── data/
    ├── TR0/                  # Training data (.st files)
    ├── TS0/                  # Test data
    └── VL0/                  # Validation data
```

## Examples

### Export FASTA from multiple directories

```bash
# Export from TR0
python scripts/export_tr0_fasta.py \
  --data_dir data/TR0 \
  --max_len 300 \
  --out_fasta tr0.fasta

# Export from TS0
python scripts/export_tr0_fasta.py \
  --data_dir data/TS0 \
  --max_len 300 \
  --out_fasta ts0.fasta
```

### Train with different split ratios

```bash
# 70% train, 15% val, 15% test
python main.py \
  --clstr_path tr0_cdhit95.clstr \
  --train_frac 0.7 \
  --val_frac 0.15
```

### Reproduce exact split from saved configuration

```bash
# First run - save split
python main.py --clstr_path tr0.clstr --split_out my_split.json

# Later - the split configuration is automatically used if you
# provide the same clstr_path and split_seed
python main.py --clstr_path tr0.clstr --split_seed 42
```

## Troubleshooting

### "No valid sequences found"

Check that:
- Your data directory path is correct
- Files have `.st` or `.dbn` extension
- Files contain `#Name:` lines
- Sequences are ≤ MAX_LEN

### "Cluster file not found"

Ensure:
- You've run the FASTA export script first
- You've run CD-HIT to generate the `.clstr` file
- The path to `.clstr` is correct

### Training with random split by mistake

If `--clstr_path` is not provided or the file doesn't exist, training falls back to random split with a warning message.

## License

[Add your license here]

## Citation

[Add citation information here]

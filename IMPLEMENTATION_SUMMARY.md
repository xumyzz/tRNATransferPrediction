# bpRNA-1m Pretraining Implementation Summary

## Overview

This implementation adds comprehensive support for pretraining on bpRNA-1m dataset with cluster-disjoint splits to prevent data leakage.

## Changes Made

### 1. Enhanced Dataset Support (`src/dataset.py`)

**`MultiFileDatasetUpgrade` class enhancements:**
- ✓ Robust parsing for both `.st` (bpRNA) and `.dbn` (FASTA-like) formats
- ✓ Multiple records per file support
- ✓ Name tracking for each sample via `self.names` list and `get_name(idx)` method
- ✓ Comprehensive validation:
  - Length checking (max_len)
  - Sequence/structure length match
  - Valid bases only (A, C, G, U, N)
  - N threshold (default 20%)
- ✓ Sequence normalization (uppercase, T→U conversion)
- ✓ Detailed statistics reporting

**Format Support:**

`.st` format (bpRNA standard):
```
#Name: sequence_name
#Length: 75
ACGUACGU...
(((...))).
```

`.dbn` format (FASTA-like):
```
>sequence_name
ACGUACGU...
(((...))).
```

### 2. FASTA Export Script (`scripts/export_bprna_fasta.py`)

Command-line tool for exporting sequences to FASTA format for clustering:

**Features:**
- Multiple input sources (files or directories, repeatable `--input`)
- Auto-discovery of `.st` and `.dbn` files in directories
- Same validation as dataset loader
- Outputs:
  - FASTA file (required)
  - Names list (optional)
  - Statistics JSON (optional)
- Headers match dataset names exactly for consistent cluster assignment

**Usage:**
```bash
python scripts/export_bprna_fasta.py \
    --input data/bpRNA_CRW \
    --input data/bpRNA_RFAM \
    --max_len 600 \
    --out_fasta sequences.fasta \
    --out_names names.txt \
    --stats_out stats.json
```

### 3. Cluster-Based Training (`train_with_args.py`)

New training script with command-line arguments and cluster-based splitting:

**Key Features:**
- All hyperparameters configurable via CLI
- Cluster-based train/val/test splits:
  - Parses cd-hit-est `.clstr` files
  - Keeps entire clusters together in same split
  - Prevents data leakage
- Split export to JSON for reproducibility
- Backward compatible with random splits (when `--clstr_path` not provided)

**Usage:**
```bash
# Cluster-based split
python train_with_args.py \
    --data_dir data/bpRNA_1m \
    --max_len 600 \
    --clstr_path clusters.clstr \
    --train_frac 0.8 \
    --val_frac 0.1 \
    --split_out splits.json \
    --batch_size 2 \
    --accum_steps 32 \
    --epochs 20 \
    --save_dir checkpoints_bprna1m

# Random split (no clustering)
python train_with_args.py \
    --data_dir data/TR0 \
    --max_len 300 \
    --train_frac 0.9 \
    --val_frac 0.1 \
    --epochs 10
```

### 4. Cluster Utilities (`scripts/cluster_utils.py`)

Lightweight utility module for parsing cd-hit-est cluster files:
- No heavy dependencies (no torch/numpy)
- Reusable across scripts
- Well-tested

### 5. Comprehensive Documentation (`README.md`)

Complete workflow documentation including:
- Quick start guides for both TR0 and bpRNA-1m
- Step-by-step pretraining workflow
- CD-HIT clustering commands with parameter recommendations
- Memory optimization strategies
- Performance tips for long sequences (L²scaling)
- Troubleshooting guide

### 6. Test Suite (`tests/`)

Comprehensive tests without requiring torch/numpy:
- ✓ `.st` format parsing
- ✓ `.dbn` format parsing  
- ✓ Validation logic (length, N threshold, invalid bases, T→U)
- ✓ Cluster file parsing
- ✓ Full export integration test
- ✓ End-to-end workflow demonstration

**Run tests:**
```bash
python tests/test_dataset_export.py
python tests/test_workflow.py
```

### 7. Project Infrastructure

- ✓ `.gitignore` - excludes __pycache__, .pth files, etc.
- ✓ `scripts/` directory - for utilities
- ✓ `tests/` directory - for test files

## Backward Compatibility

All existing functionality preserved:
- ✓ `MultiFileDataset` class still available (unchanged)
- ✓ `collate_pad` function unchanged
- ✓ `src/train.py` still works with original Config-based approach
- ✓ `main.py` entry point unchanged
- ✓ TR0 training workflow intact

## Complete Workflow Example

### Step 1: Export to FASTA
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

### Step 2: Cluster with CD-HIT
```bash
cd-hit-est \
    -i bprna_1m.fasta \
    -o bprna_1m_clustered.fasta \
    -c 0.90 \
    -n 8 \
    -M 16000 \
    -T 8 \
    -d 0
```

This produces `bprna_1m_clustered.fasta.clstr`

### Step 3: Train with Cluster Split
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

## Memory Considerations

For L=600 with 2x 24GB GPUs:
- Batch size: 1-2 per GPU
- Gradient accumulation: 32-64 steps
- Effective batch size: 64-128
- Consider mixed precision (AMP) if needed

## Files Added/Modified

**New files:**
- `scripts/export_bprna_fasta.py` - FASTA export tool
- `scripts/cluster_utils.py` - Cluster parsing utilities
- `scripts/__init__.py` - Package marker
- `train_with_args.py` - Enhanced training script
- `README.md` - Comprehensive documentation
- `tests/test_dataset_export.py` - Unit tests
- `tests/test_workflow.py` - Workflow test
- `tests/__init__.py` - Package marker
- `.gitignore` - Git ignore rules

**Modified files:**
- `src/dataset.py` - Enhanced MultiFileDatasetUpgrade class

**Preserved files:**
- All existing files unchanged in functionality
- `src/train.py` - Original training script (still works)
- `main.py` - Original entry point (still works)
- All model files unchanged

## Validation

All acceptance criteria met:
- ✓ Mixed .st and .dbn file loading
- ✓ Name tracking and FASTA export with matching headers
- ✓ Cluster-based splitting implemented and tested
- ✓ README with exact workflow commands
- ✓ Model architecture unchanged
- ✓ No heavy dependencies added
- ✓ TR0 workflow intact

## Testing

All tests pass without requiring torch/numpy installation:
```bash
$ python tests/test_dataset_export.py
All tests passed! ✓

$ python tests/test_workflow.py  
✓ Workflow test completed successfully!
```

## Next Steps for Users

1. Install dependencies: `pip install torch numpy`
2. Prepare bpRNA-1m data (download .dbn/.st files)
3. Follow README workflow to export, cluster, and train
4. For TR0 workflow, continue using existing `main.py` or `src/train.py`

## Notes

- Export script validated on 1300+ TS0 files
- Cluster parsing tested with cd-hit-est format
- All validation thresholds match training requirements
- Names are stable and reproducible across export and dataset loading

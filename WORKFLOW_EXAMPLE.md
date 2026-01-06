# Cluster-Based Training Workflow Example

This document provides a step-by-step example of using cluster-based splitting to prevent data leakage.

## Prerequisites

- CD-HIT installed (for clustering)
- Python with PyTorch and NumPy

## Complete Workflow

### Step 1: Export FASTA from TR0 Data

```bash
# Export sequences from TR0 directory
python scripts/export_tr0_fasta.py \
  --data_dir data/TR0 \
  --max_len 300 \
  --out_fasta tr0_sequences.fasta \
  --out_names tr0_names.txt

# Expected output:
# 🧐 正在扫描 10814 个文件 (MaxLen=300)...
# ✅ 有效序列: 10026
# ✅ 写入 10026 条序列到 tr0_sequences.fasta
```

### Step 2: Run CD-HIT Clustering

```bash
# Cluster sequences at 95% identity
cd-hit-est \
  -i tr0_sequences.fasta \
  -o tr0_cdhit95 \
  -c 0.95 \
  -n 10 \
  -d 0 \
  -M 0 \
  -T 0

# This creates:
# - tr0_cdhit95 (representative sequences)
# - tr0_cdhit95.clstr (cluster file - this is what we need!)
```

**CD-HIT Parameters Explained:**
- `-c 0.95`: 95% sequence identity threshold
- `-n 10`: Word length (use 10 for threshold ≥0.9, 8 for 0.88-0.9, etc.)
- `-d 0`: Full sequence names in .clstr file (important!)
- `-M 0`: Unlimited memory usage
- `-T 0`: Use all available CPU threads

### Step 3: Train with Cluster-Based Split

```bash
# Train with cluster-based splitting
python main.py \
  --clstr_path tr0_cdhit95.clstr \
  --split_seed 42 \
  --train_frac 0.8 \
  --val_frac 0.1 \
  --split_out splits.json

# Expected output:
# 📊 解析 tr0_cdhit95.clstr:
#   - 找到 10026 个序列
#   - 分布在 XXXX 个簇中
# 
# ==================================================
# 📊 聚类分割统计:
#   训练集: XXXX 样本 (XXXX 个簇)
#   验证集: XXXX 样本 (XXXX 个簇)
#   测试集: XXXX 样本 (XXXX 个簇)
# ==================================================
```

### Step 4: Verify Split Integrity (Optional)

```bash
# Check that the split was saved correctly
cat splits.json | head -20
```

The JSON file contains:
- `train_indices`: List of training sample indices
- `val_indices`: List of validation sample indices
- `test_indices`: List of test sample indices
- `metadata`: Information about the split (seed, fractions, etc.)

## Alternative: Training Without Clustering (Random Split)

For comparison or when clustering is not needed:

```bash
# Train with random split (default behavior)
python main.py
```

This uses the original random_split behavior with 90% train, 10% validation.

## Tips and Best Practices

### Choosing Identity Threshold

- **0.95 (95%)**: Recommended for tRNA (removes very similar sequences)
- **0.90 (90%)**: More aggressive clustering (removes moderately similar)
- **0.80 (80%)**: Very aggressive (only for very diverse datasets)

### Handling Warnings

If you see:
```
⚠️  WARNING: XXX 个样本未在聚类文件中找到！
   这占总数据集的 XX.X%
```

This means some samples in the training data weren't in the FASTA export. Common causes:
1. Different `max_len` used in export vs training
2. Different data directory paths
3. Files added/removed between export and training

**Fix:** Re-run the FASTA export with the same parameters as training.

### Split Fractions

Common configurations:
- **80/10/10**: Standard (80% train, 10% val, 10% test)
- **70/15/15**: More validation/test data
- **85/10/5**: More training data, less test

Always ensure: `train_frac + val_frac ≤ 1.0` (remainder goes to test)

## Verifying No Data Leakage

After training, you can verify cluster integrity:

```bash
# Run the validation test
python scripts/test_cluster_logic.py

# All tests should pass:
# ✅ ALL PARSER TESTS PASSED
# ✅ ALL SPLIT LOGIC TESTS PASSED
```

## Troubleshooting

### Problem: "No valid sequences found"

**Solution:** Check that:
- Data directory path is correct
- Files have `.st` extension
- `--max_len` matches your data

### Problem: "Cluster file not found"

**Solution:** 
- Ensure CD-HIT completed successfully
- Check the `.clstr` file path is correct
- Verify the file exists: `ls -l tr0_cdhit95.clstr`

### Problem: "Too many unknown samples" warning

**Solution:**
- Re-run FASTA export with same parameters as training
- Check that both use same `--max_len`
- Verify same data directory

## Example: Full Pipeline Script

```bash
#!/bin/bash
# complete_pipeline.sh

# 1. Export FASTA
echo "Step 1: Exporting FASTA..."
python scripts/export_tr0_fasta.py \
  --data_dir data/TR0 \
  --max_len 300 \
  --out_fasta tr0.fasta \
  --out_names tr0_names.txt

# 2. Cluster with CD-HIT
echo "Step 2: Clustering with CD-HIT..."
cd-hit-est -i tr0.fasta -o tr0_cdhit95 -c 0.95 -n 10 -d 0 -M 0 -T 0

# 3. Train with cluster-based split
echo "Step 3: Training with cluster-based split..."
python main.py \
  --clstr_path tr0_cdhit95.clstr \
  --split_seed 42 \
  --train_frac 0.8 \
  --val_frac 0.1 \
  --split_out splits.json

echo "Done!"
```

Make it executable and run:
```bash
chmod +x complete_pipeline.sh
./complete_pipeline.sh
```

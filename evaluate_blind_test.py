import os
import sys
import json
import random
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.utils import compute_masked_loss
from src.dataset import MultiFileDatasetUpgrade, collate_pad
from src.model import SpotRNA_LSTM_Refined
from scripts.cluster_utils import parse_cd_hit_clusters

# UFold-style postprocess metric
from src.metrics import calculate_f1_postprocess_ufold

def dotbracket_to_matrix(seq, dotbracket):
    """将点括号序列转化为真实的 2D 配对矩阵 (Ground Truth)"""
    L = len(seq)
    matrix = np.zeros((L, L), dtype=np.float32)
    stack = []
    
    for i, char in enumerate(dotbracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                # 矩阵是对称的
                matrix[i, j] = 1.0
                matrix[j, i] = 1.0
    return matrix

def encode_sequence(seq):
    """将 ACGU 序列转化为模型认识的 3D One-Hot Tensor: [Batch=1, Length, 4]"""
    # 纯正的 4 维字典: A=0, C=1, G=2, U=3
    vocab = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    L = len(seq)
    
    # 初始化一个全 0 的 3D 张量，形状为 (1, L, 4)
    one_hot = np.zeros((1, L, 4), dtype=np.float32)
    
    for i, char in enumerate(seq):
        if char in vocab:
            idx = vocab[char]
            one_hot[0, i, idx] = 1.0
        # 如果是 N 或其他奇怪字符，什么都不做，直接保留全 0 即可
        
    return torch.tensor(one_hot)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔬 正在使用 {device} 进行盲测集推断...")

    # 1. 初始化模型并加载权重
    config = Config()
    model = SpotRNA_LSTM_Refined(config).to(device)
    
    if not os.path.exists(args.weight_path):
        raise FileNotFoundError(f"找不到权重文件: {args.weight_path}")
        
    model.load_state_dict(torch.load(args.weight_path, map_location=device))
    model.eval()
    print("✅ 模型权重加载成功！")

    # 2. 读取盲测集数据
    with open(args.data_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    total_TP, total_FP, total_FN = 0, 0, 0
    seq_count = 0

    print(f"🚀 开始评估 pp_offset = {args.pp_offset} ...\n")
    print("-" * 50)

    # 3. 逐条推断与严格测算
    with torch.no_grad():
        for i in range(0, len(lines), 3):
            header = lines[i]
            seq = lines[i+1]
            dotbracket = lines[i+2]
            
            L = len(seq)
            
            # 准备 Input 和 Ground Truth
            seq_tensor = encode_sequence(seq).to(device)
            mask_tensor = torch.ones((1, L), dtype=torch.float32).to(device) # Batch=1, 全有效
            gt_matrix = torch.tensor(dotbracket_to_matrix(seq, dotbracket)).to(device)

            # 模型前向传播
            logits = model(seq_tensor, mask=mask_tensor)
            probs = torch.sigmoid(logits).squeeze(0) # 剥离 Batch 维度，变成 [L, L]

            # 强制对称化物理约束 (i配j 等于 j配i)
            probs = (probs + probs.t()) / 2.0

            # 阈值截断 (UFold 逻辑)
            preds = (probs > args.pp_offset).float()

            # 统计混淆矩阵 (排除对角线和下三角，避免重复计算)
            # 使用 torch.triu 提取上三角矩阵
            preds_triu = torch.triu(preds, diagonal=1)
            gt_triu = torch.triu(gt_matrix, diagonal=1)

            TP = (preds_triu * gt_triu).sum().item()
            FP = (preds_triu * (1 - gt_triu)).sum().item()
            FN = ((1 - preds_triu) * gt_triu).sum().item()

            total_TP += TP
            total_FP += FP
            total_FN += FN
            seq_count += 1

            # 打印单条结果
            p = TP / (TP + FP + 1e-8)
            r = TP / (TP + FN + 1e-8)
            f1 = 2 * p * r / (p + r + 1e-8)
            print(f"[{seq_count}] {header} | L={L} | P: {p:.4f}  R: {r:.4f}  F1: {f1:.4f}")

    print("-" * 50)
    
    # 4. 计算全局总指标 (Micro-F1，最客观的科学评价标准)
    final_P = total_TP / (total_TP + total_FP + 1e-8)
    final_R = total_TP / (total_TP + total_FN + 1e-8)
    final_F1 = 2 * final_P * final_R / (final_P + final_R + 1e-8)

    print("\n================ 👑 盲测集最终成绩 ================")
    print(f"测试集序列总数: {seq_count}")
    print(f"全局 Precision (精确率): {final_P:.4f}")
    print(f"全局 Recall    (召回率): {final_R:.4f}")
    print(f"全局 F1 Score  (F1分数): {final_F1:.4f}")
    print("===================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="刚才清洗好的 clean_blind_test.txt")
    parser.add_argument("--weight_path", type=str, required=True, help="你的 model_best.pth 路径")
    parser.add_argument("--pp_offset", type=float, default=0.5, help="及格线阈值，建议用 0.5")
    
    args = parser.parse_args()
    main(args)
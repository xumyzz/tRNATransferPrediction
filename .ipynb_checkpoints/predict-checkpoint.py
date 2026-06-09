import torch
import numpy as np
import RNA  # 导入 ViennaRNA 包
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

# 1. 序列转 One-hot 的辅助函数
def seq_to_one_hot(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
    seq = seq.upper()
    L = len(seq)
    one_hot = np.zeros((L, 4), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            one_hot[i, mapping[char]] = 1.0
    return torch.tensor(one_hot).unsqueeze(0)

def prob_to_energy_bonus(prob, min_prob=0.5, max_bonus=-3.0):
    """
    将深度学习输出的概率转换为伪自由能奖励 (Pseudo-energy perturbation)
    - prob: 模型输出的概率 (0~1)
    - min_prob: 门槛值，低于此值不给能量奖励
    - max_bonus: 满分概率(1.0)能获得的负能量值 (越小代表越稳定，引导作用越强)
    """
    if prob < min_prob:
        return 0.0
    # 线性插值映射：让概率越高的碱基对，获得的能量减免越多
    bonus = max_bonus * ((prob - min_prob) / (1.0 - min_prob))
    return bonus


# =========================================================================
# 2. 【核心】基于伪自由能的软约束预测流程
# =========================================================================
def predict_chimeric_rna_soft(sequence, model, min_prob=0.5, max_bonus=-3.0):
    print(f"-> 正在预测序列长度: {len(sequence)} nt")
    
    # 步骤A: DL 前向传播获取全局配对概率矩阵
    x = seq_to_one_hot(sequence)
    with torch.no_grad():
        logits = model(x) 
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy() # (L, L)
    
    # 步骤B: 初始化 RNAfold 物理折叠对象
    fc = RNA.fold_compound(sequence)
    
    L = len(sequence)
    added_count = 0
    
    # 步骤C: 遍历概率矩阵，注入软约束 (伪自由能扰动)
    for i in range(L):
        # 物理规则：发卡环最少需要3个未配对碱基，所以 j 至少从 i+4 开始
        for j in range(i + 4, L):
            p = probs[i, j]
            if p > min_prob:
                # 计算这对其应得的能量奖励
                bonus = prob_to_energy_bonus(p, min_prob, max_bonus)
                # API 说明：sc_add_bp(i, j, energy, options) 
                # 强转为 python 原生 float 防止 numpy 类型报错，并指定该约束用于 MFE 计算
                fc.sc_add_bp(int(i + 1), int(j + 1), float(bonus), RNA.OPTION_MFE)
                added_count += 1
                
    print(f"-> DL模型为 {added_count} 对潜在碱基配对施加了能量扰动奖励。")

    # 步骤D: 计算融合了 DL 奖励的 MFE 结构
    mfe_struct, pseudo_mfe = fc.mfe()
    
    # 步骤E: 重新计算该结构在自然状态下（去掉人为 DL 奖励后）的真实物理能量
    real_energy = RNA.energy_of_struct(sequence, mfe_struct)
    
    return mfe_struct, real_energy, pseudo_mfe


# =========================================================================
# 3. 运行入口与测试
# =========================================================================
if __name__ == "__main__":
    # 配置必须与你重新训练的网络参数完全一致
    class Config:
        HIDDEN_DIM = 64
        RESNET_LAYERS = 8
        LSTM_HIDDEN = 64

    config = Config()
    
    # 实例化模型
    model = SpotRNA_LSTM_Refined(config)

    # ================= 注意：加载新权重的步骤 =================
    # 你必须先用带 Attention 的架构完成训练，然后放开下面这两行的注释填入路径。
    # 否则当前没加载权重的模型是瞎猜的，软约束也会加错地方。
    # 
    weight_path = "/root/autodl-tmp/myPredicProject/weight/Attention_Model/model_best.pth"
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    
    model.eval() 

    # 待预测的嵌合序列
    my_chimeric_seq = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUUUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCAUGUGGUCGACAGGUGUAUGAAGACUGUCACGGGCAAGUUGCGGAA"
    
    print("\n--- 开始执行软约束预测 (Soft Constraints) ---")
    struct, real_energy, pseudo_mfe = predict_chimeric_rna_soft(
        my_chimeric_seq, 
        model, 
        min_prob=0.5,    # 门槛：概率大于 0.5 才会触发能量奖励
        max_bonus=-3.0   # 上限：概率达到 1.0 时给予的最高能量奖励 (-3.0 kcal/mol)
    )
    
    print("\n================ 预测最终结果 ================")
    print(f"序列: {my_chimeric_seq}")
    print(f"结构: {struct}")
    print(f"包含DL奖励的伪能量: {pseudo_mfe:.2f} kcal/mol")
    print(f"纯物理真实能量(客观评估用): {real_energy:.2f} kcal/mol")


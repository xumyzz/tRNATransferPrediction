import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
import RNA

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.utils import compute_masked_loss
from src.dataset import MultiFileDatasetUpgrade, collate_pad
from src.model import SpotRNA_LSTM_Refined
from src.model import SpotRNA_LSTM_Refined_Attention
from scripts.cluster_utils import parse_cd_hit_clusters

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
    # 这是原来的正向奖励函数（保持不变）
    if prob < min_prob:
        return 0.0
    return max_bonus * ((prob - min_prob) / (1.0 - min_prob))

def predict_chimeric_rna_soft(sequence, model, min_prob=0.5, max_bonus=-3.0, max_penalty=10.0, penalty_threshold=0.1):
    """
    增加了 max_penalty 和 penalty_threshold 参数：
    - min_prob: 大于这个概率的，给负能量(奖励)
    - penalty_threshold: 小于这个概率的，给巨大的正能量(惩罚，阻止配对)
    - max_penalty: 惩罚能量大小 (默认 +10.0 kcal/mol，极大阻力)
    """
    device = next(model.parameters()).device 
    x = seq_to_one_hot(sequence).to(device) 
    
    with torch.no_grad():
        logits = model(x) 
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    
    fc = RNA.fold_compound(sequence)
    L = len(sequence)
    
    for i in range(L):
        for j in range(i + 4, L):
            p = probs[i, j]
            
            # 情况1：高置信度配对 -> 给予奖励 (负能量)
            if p > min_prob:
                bonus = prob_to_energy_bonus(p, min_prob, max_bonus)
                fc.sc_add_bp(int(i + 1), int(j + 1), float(bonus), RNA.OPTION_MFE)
                
            # 情况2：极低置信度配对 -> 给予重罚 (正能量)
            # 这能阻止物理引擎在这里跨域胡乱配对
            elif p < penalty_threshold:
                # 给物理引擎施加强大的能量壁垒
                fc.sc_add_bp(int(i + 1), int(j + 1), float(max_penalty), RNA.OPTION_MFE)

    mfe_struct, pseudo_mfe = fc.mfe()
    real_energy = RNA.energy_of_struct(sequence, mfe_struct)
    
    return mfe_struct, real_energy, pseudo_mfe

if __name__ == "__main__":
    # ================= 1. 初始化模型 =================
    class Config:
        HIDDEN_DIM = 64
        RESNET_LAYERS = 8
        LSTM_HIDDEN = 64

    print("正在加载模型权重...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpotRNA_LSTM_Refined(Config()).to(device)
    
    # # 填入你最新训练的 Attention 权重路径
    # weight_path = "/root/autodl-tmp/myPredicProject/weight/Attention_Model/model_best.pth" 

    #老模型权重
    weight_path = "/root/autodl-tmp/myPredicProject/weight/tRNA_Finetune/model_best.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # ================= 2. 读取数据 =================
    print("正在读取 tRNA 和 miRNA 数据...")
    df_trna = pd.read_csv("/root/autodl-tmp/myPredicProject/tRNA_mRNA_Binding_data/tRNA_list.csv")
    df_mirna = pd.read_csv("/root/autodl-tmp/myPredicProject/tRNA_mRNA_Binding_data/miRNA_list.csv")
    
    # 设置嵌合体连接的 Linker
    linker_seq = "AAAAA"

    results = []
    total_tasks = len(df_trna) * len(df_mirna)
    count = 1

    # ================= 3. 批量预测 =================
    print(f"开始批量预测，总计 {total_tasks} 条序列...")
    for _, row_t in df_trna.iterrows():
        for _, row_m in df_mirna.iterrows():
            t_name, t_seq = row_t['Name'], row_t['Sequence']
            m_name, m_seq = row_m['Name'], row_m['Sequence']
            
            # 拼接: tRNA + Linker + miRNA
            chimeric_name = f"{t_name}_{m_name}"
            chimeric_seq = t_seq + linker_seq + m_seq
            
            print(f"[{count}/{total_tasks}] 正在预测: {chimeric_name} (长度: {len(chimeric_seq)})")
            
                # 引入非对称惩罚。
            # min_prob=0.6 稍微提高奖励门槛，确保只有极其确定的配对才被奖励
            # penalty_threshold=0.1 当概率低于10%时，绝对不许配对
            # max_penalty=10.0 巨大的能量墙，截断串联拉伸
            struct, real_energy, pseudo_energy = predict_chimeric_rna_soft(
                chimeric_seq, model, min_prob=0.6, max_bonus=-3.0, penalty_threshold=0.1, max_penalty=10.0
            )
            
            # 保存到结果字典
            results.append({
                "Chimera_Name": chimeric_name,
                "tRNA_Part": t_name,
                "miRNA_Part": m_name,
                "Linker": linker_seq,
                "Full_Sequence": chimeric_seq,
                "Predicted_Structure": struct,
                "Pseudo_Energy (kcal/mol)": round(pseudo_energy, 2),
                "Real_Energy (kcal/mol)": round(real_energy, 2)
            })
            count += 1

    # ================= 4. 导出 Excel =================
    df_results = pd.DataFrame(results)
    output_excel = "/root/autodl-tmp/myPredicProject/tRNA_mRNA_Binding_data/Chimeric_RNA_Predictions1.xlsx"
    df_results.to_excel(output_excel, index=False)
    
    print(f"\n✅ 批量预测完成！所有结果已保存至: {output_excel}")
import os
import numpy as np
import pandas as pd
import RNA
import torch

from src.model import SpotRNA_LSTM_Refined_BPPM # 注意：导入的是我们新建的 BPPM 模型类

def seq_to_one_hot(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
    seq = seq.upper()
    L = len(seq)
    one_hot = np.zeros((L, 4), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            one_hot[i, mapping[char]] = 1.0
    return torch.tensor(one_hot).unsqueeze(0)

# 【核心新增】实时生成 BPPM 通道特征的函数
def get_bppm_feature(seq):
    L = len(seq)
    safe_seq = seq.replace('N', 'A') 
    fc = RNA.fold_compound(safe_seq)
    
    try:
        fc.pf()  
        bpp = fc.bpp()  
    except Exception as e:
        return torch.zeros((1, 1, L, L), dtype=torch.float32)
        
    matrix = np.zeros((L, L), dtype=np.float32)
    for i in range(1, L + 1):
        for j in range(i + 1, L + 1):
            prob = bpp[i][j]
            matrix[i-1, j-1] = prob
            matrix[j-1, i-1] = prob
            
    # 增加 batch 和 channel 维度 -> (1, 1, L, L)
    return torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def prob_to_energy_bonus(prob, min_prob=0.5, max_bonus=-3.0):
    if prob < min_prob:
        return 0.0
    return max_bonus * ((prob - min_prob) / (1.0 - min_prob))

def predict_chimeric_rna_soft_bppm(sequence, model, min_prob=0.5, max_bonus=-3.0):
    device = next(model.parameters()).device 
    
    # 1. 准备双通道输入
    x = seq_to_one_hot(sequence).to(device) 
    bppm_feat = get_bppm_feature(sequence).to(device)
    
    # 2. 模型推理 (传入两个特征)
    with torch.no_grad():
        logits = model(x, bppm=bppm_feat) 
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    
    # 3. 回归经典的“局部软约束奖励”，去掉惩罚
    fc = RNA.fold_compound(sequence)
    L = len(sequence)
    
    for i in range(L):
        for j in range(i + 4, L):
            p = probs[i, j]
            if p > min_prob:
                bonus = prob_to_energy_bonus(p, min_prob, max_bonus)
                fc.sc_add_bp(int(i + 1), int(j + 1), float(bonus), RNA.OPTION_MFE)

    mfe_struct, pseudo_mfe = fc.mfe()
    real_energy = RNA.energy_of_struct(sequence, mfe_struct)
    
    return mfe_struct, real_energy, pseudo_mfe

if __name__ == "__main__":
    # ================= 1. 初始化新模型 =================
    class Config:
        HIDDEN_DIM = 64
        RESNET_LAYERS = 8
        LSTM_HIDDEN = 64

    print("正在初始化带有 BPPM 通道的全新模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 【注意】这里实例化的是新类
    model = SpotRNA_LSTM_Refined_BPPM(Config()).to(device)
    
    # 【填入你即将重新训练出来的、包含了 BPPM 的新权重路径】
    # weight_path = "/root/autodl-tmp/myPredicProject/weight/BPPM_Model/model_best.pth" 
    # model.load_state_dict(torch.load(weight_path, map_location=device))
    
    model.eval()

    # ================= 2. 读取数据 =================
    print("正在读取 tRNA 和 miRNA 数据...")
    df_trna = pd.read_csv("/root/autodl-tmp/myPredicProject/tRNA_mRNA_Binding_data/tRNA_list.csv")
    df_mirna = pd.read_csv("/root/autodl-tmp/myPredicProject/tRNA_mRNA_Binding_data/miRNA_list.csv")
    
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
            
            chimeric_name = f"{t_name}_{m_name}"
            chimeric_seq = t_seq + linker_seq + m_seq
            
            print(f"[{count}/{total_tasks}] 正在预测: {chimeric_name}")
            
            # 使用新版的预测函数，恢复经典的 0.5 门槛和 -3.0 奖励
            struct, real_energy, pseudo_energy = predict_chimeric_rna_soft_bppm(
                chimeric_seq, model, min_prob=0.5, max_bonus=-3.0
            )
            
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
    output_excel = "/root/autodl-tmp/myPredicProject/tRNA_mRNA_Binding_data/Chimeric_RNA_Predictions_BPPM_Version.xlsx"
    df_results.to_excel(output_excel, index=False)
    
    print(f"\n✅ 批量预测完成！所有结果已保存至: {output_excel}")
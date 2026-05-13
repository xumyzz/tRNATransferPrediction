import os
import argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.dataset import MultiFileDatasetUpgrade, collate_pad
from src.model import SpotRNA_LSTM_Refined as ModelClass
from src.metrics import calculate_f1_postprocess_ufold

def get_binary_prediction(logits, mask, offset=0.5, min_loop=4):
    """把模型的原始输出转化为最终的二值化配��矩阵"""
    # 1. 转换为概率
    prob = torch.sigmoid(logits)
    # 2. 阈值截断
    pred = (prob > offset).float()
    
    # 3. 施加生物学约束：消除对角线及极短环（|i - j| < min_loop）的配对
    L = pred.shape[-1]
    idx = torch.arange(L, device=pred.device)
    dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    loop_mask = (dist >= min_loop).float()
    
    # 4. 把 1D mask (B, L) 变成 2D mask (B, L, L)，消除 padding 区域的乱猜
    if mask.dim() == 2:
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
    else:
        mask_2d = mask
        
    pred = pred * loop_mask * mask_2d
    return pred

def plot_contact_map(true_mat, pred_mat, seq_name, seq_len, f1, p, r, save_dir):
    """绘制高逼格学术接触图 (右上真实，左下预测)"""
    # 截取真实的有效长度（去除 Padding 的空白区域）
    true_mat = true_mat[:seq_len, :seq_len]
    pred_mat = pred_mat[:seq_len, :seq_len]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 画对角线（参考线）
    ax.plot([0, seq_len], [0, seq_len], color='gray', linestyle='--', alpha=0.5)
    
    # === 提取坐标 ===
    # 右上角 (Upper Triangle) 画真实的 Ground Truth
    true_rows, true_cols = np.where(np.triu(true_mat, k=1) == 1)
    # 左下角 (Lower Triangle) 画模型的 Prediction
    pred_rows, pred_cols = np.where(np.tril(pred_mat, k=-1) == 1)
    
    # 画真实的点 (绿色, 稍大)
    ax.scatter(true_cols, true_rows, s=60, marker='s', color='#2ca02c', label='Ground Truth (True)', alpha=0.8)
    
    # 画预测的点 (红色, 稍小，在左下角呈现对称的接触图)
    ax.scatter(pred_cols, pred_rows, s=60, marker='s', color='#d62728', label='Prediction (Model)', alpha=0.8)

    # 美化图表
    ax.set_title(f"Contact Map: {seq_name}\nL={seq_len} | F1: {f1:.4f} | Prec: {p:.4f} | Rec: {r:.4f}", 
                 fontsize=14, pad=15, fontweight='bold')
    ax.set_xlim(0, seq_len)
    ax.set_ylim(seq_len, 0) # Y轴反转
    ax.set_xlabel("Sequence Index (j)", fontsize=12)
    ax.set_ylabel("Sequence Index (i)", fontsize=12)
    
    # 设置刻度
    ticks = np.arange(0, seq_len+1, step=10)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.5)
    
    # 图例
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    
    # 保存图片
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{seq_name}_contact_map.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📸 已生成图片: {save_path}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎨 开始生成可视化接触图...")
    
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = MultiFileDatasetUpgrade(args.data_dir, max_len=args.max_len)
    #42保证客可复现
    # random.seed(42)
    sample_indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    sample_ds = Subset(dataset, sample_indices)
    
    loader = DataLoader(sample_ds, batch_size=1, shuffle=False, collate_fn=collate_pad)

    config = Config()
    config.RESNET_LAYERS = args.resnet_layers
    config.HIDDEN_DIM = args.hidden_dim
    config.LSTM_HIDDEN = args.lstm_hidden
    config.DEVICE = device

    model = ModelClass(config).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        for i, (seqs, labels, masks) in enumerate(loader):
            seqs = seqs.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            # 🚀 修复点：对于 (B, L) 形状的 mask，直接在第 0 维度求和获取长度
            seq_len = int(masks[0].sum().item())
            
            seq_name = f"tRNA_Sample_{i+1}"
            
            logits = model(seqs, mask=masks)
            
            f1, p, r = calculate_f1_postprocess_ufold(
                logits=logits, labels=labels, masks=masks, seqs=seqs, 
                offset=args.pp_offset, min_loop=args.pp_min_loop
            )
            
            pred_matrix = get_binary_prediction(logits, masks, args.pp_offset, args.pp_min_loop)
            
            true_mat_np = labels[0].cpu().numpy()
            pred_mat_np = pred_matrix[0].cpu().numpy()
            
            plot_contact_map(
                true_mat=true_mat_np, 
                pred_mat=pred_mat_np, 
                seq_name=seq_name, 
                seq_len=seq_len, 
                f1=f1, p=p, r=r, 
                save_dir=args.save_dir
            )

    print("\n🎉 所有绘图完成！请前往文件夹查看。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='visualizations')
    parser.add_argument('--num_samples', type=int, default=5)
    
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--resnet_layers', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lstm_hidden', type=int, default=64)
    parser.add_argument('--pp_offset', type=float, default=0.5)
    parser.add_argument('--pp_min_loop', type=int, default=4)
    
    args = parser.parse_args()
    main(args)
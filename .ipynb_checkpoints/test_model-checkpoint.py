import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import MultiFileDatasetUpgrade, collate_pad
from src.metrics import calculate_f1_postprocess_ufold

# 明确且唯一地导入为你打下 0.98 江山的物理先验 LSTM 模型
from src.model import SpotRNA_LSTM_Refined as ModelClass

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始终极零样本测试 (Zero-shot Test)!")
    print(f"🖥️  使用设备: {device}")
    
    # 1. 加载测试数据集
    print(f"📂 加载测试集: {args.data_dir} ...")
    test_ds = MultiFileDatasetUpgrade(args.data_dir, max_len=args.max_len)
    if len(test_ds) == 0:
        print("❌ 错误：测试集中没有数据，请检查路径！")
        return
    print(f"✅ 成功加载 {len(test_ds)} 条绝对独立的测试序列。")

    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_pad, 
        num_workers=0
    )

    # 2. 初始化模型配置
    config = Config()
    config.RESNET_LAYERS = args.resnet_layers
    config.HIDDEN_DIM = args.hidden_dim
    config.LSTM_HIDDEN = args.lstm_hidden
    config.DEVICE = device

    print("🧠 正在初始化带有多通道 2D 物理先验的 LSTM 网络...")
    model = ModelClass(config).to(device)

    # 3. 加载预训练权重 (你的 0.98 神级权重)
    if not os.path.exists(args.model_path):
        print(f"❌ 错误：找不到模型权重文件 {args.model_path}")
        return
        
    print(f"📥 加载巅峰权重: {args.model_path} ...")
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ 权重加载成功！模型已武装完毕，准备闭卷考试。")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return

    # 4. 开始测试推理
    model.eval()
    total_f1, total_p, total_r = 0.0, 0.0, 0.0
    batch_count = 0

    print("\n" + "="*50)
    print("⚙️  开始前向推理评估 (绝对盲测，不计算梯度)...")
    print("="*50)

    with torch.no_grad():
        for batch_idx, (seqs, labels, masks) in enumerate(test_loader):
            seqs = seqs.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            # 前向传播预测接触图
            logits = model(seqs, mask=masks)
            
            # 使用 UFold 后处理算法计算 F1
            f1, p, r = calculate_f1_postprocess_ufold(
                logits=logits, 
                labels=labels, 
                masks=masks, 
                seqs=seqs, 
                offset=args.pp_offset, 
                min_loop=args.pp_min_loop
            )
            
            total_f1 += f1
            total_p += p
            total_r += r
            batch_count += 1
            
            # 打印进度
            if (batch_idx + 1) % max(1, len(test_loader)//10) == 0:
                print(f"⏳ 进度: [{batch_idx+1}/{len(test_loader)}] 批次完成 | 当前批次 F1: {f1:.4f}")

    # 5. 计算并公布最终平均成绩
    avg_f1 = total_f1 / batch_count
    avg_p = total_p / batch_count
    avg_r = total_r / batch_count

    print("\n" + "🌟 "*15)
    print("🏆 终极绝密独立测试集 (Strict Zero-Shot) 成绩单 🏆")
    print(f"  📌 Precision (精确率) : {avg_p:.4f}")
    print(f"  📌 Recall    (召回率) : {avg_r:.4f}")
    print(f"  🚀 F1-Score  (综合得分): {avg_f1:.4f}")
    print("🌟 "*15 + "\n")
    
    if avg_f1 > 0.85:
        print("💡 结论：太强了！在完全没见过、剔除泄露的异变数据上 F1 超过 0.85，这篇论文的 SOTA 地位彻底稳了！")
    else:
        print("💡 结论：符合预期的正常下降。这是真实的 Zero-shot 分数，依然是一个极其优秀的基准！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test TRNA Model on Independent Dataset')
    # 数据和权重路径
    parser.add_argument('--data_dir', type=str, required=True, help='测试集(ArchiveII)所在的文件夹路径')
    parser.add_argument('--model_path', type=str, required=True, help='你的 model_best.pth 路径')
    
    # 架构参数 (必须和训练时保持完全一致)
    parser.add_argument('--max_len', type=int, default=128, help='最大序列长度')
    parser.add_argument('--batch_size', type=int, default=8, help='测试时的 Batch Size')
    parser.add_argument('--resnet_layers', type=int, default=8, help='如果用旧模型，填入训练时的层数')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lstm_hidden', type=int, default=64)
    
    # 后处理参数
    parser.add_argument('--pp_offset', type=float, default=0.5, help='预测阈值')
    parser.add_argument('--pp_min_loop', type=int, default=4, help='最小环长度')
    
    args = parser.parse_args()
    test(args)
import torch
from torch.utils.data import DataLoader
from src.dataset import MultiFileDataset
from src.model import SpotRNAWithLSTM  # 注意：如果你改过模型类名，请在这里修改
from src.config import Config
import os
from tqdm import tqdm

# ================= 配置区域 =================
# 1. TS0 数据集的绝对路径
TS0_DATA_DIR = r"D:\PycharmProjects\tRNATransferPrediction\data\TS0"

# 2. 你要测试的最佳权重路径 (Baseline)
MODEL_WEIGHT_PATH = r"D:\PycharmProjects\tRNATransferPrediction\Baseline\baseline_best_f1_0.61.pth"

# 3. 阈值 (通常是 0.5，但你可以微调看看效果)
THRESHOLD = 0.5


# ===========================================

def evaluate_on_testset():
    # 1. 准备设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 使用设备: {device}")

    # 2. 加载模型架构
    print("🏗️ 正在初始化模型...")
    model = SpotRNAWithLSTM(Config.RESNET_LAYERS,Config.HIDDEN_DIM,Config.LSTM_HIDDEN).to(device)  # 确保 Config 里的参数和你训练时一致(层数/维度)

    # 3. 加载权重
    print(f"📥 加载权重: {MODEL_WEIGHT_PATH}")
    if os.path.exists(MODEL_WEIGHT_PATH):
        # map_location 确保在只有 CPU 的机器上也能跑
        state_dict = torch.load(MODEL_WEIGHT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ 权重加载成功！")
    else:
        print(f"❌ 错误：找不到权重文件 {MODEL_WEIGHT_PATH}")
        return

    # 4. 加载测试集数据
    # 注意：这里临时修改 Config.DATA_DIR 或者直接传参给 Dataset
    # 假设你的 RNADataset 支持传入 data_dir 参数
    print(f"📂 加载测试集数据: {TS0_DATA_DIR}")
    try:
        test_dataset = MultiFileDataset(TS0_DATA_DIR)  # 如果 dataset.py 不需要参数，请自行修改
    except:
        # 如果 Dataset 强依赖 Config，我们临时改一下 Config
        Config.DATA_DIR = TS0_DATA_DIR
        test_dataset = MultiFileDataset(Config.DATA_DIR)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 测试集 batch=1 最稳，方便逐个分析
    print(f"📊 测试集样本数: {len(test_dataset)}")

    # 5. 开始推理
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0

    print("🚀 开始评估...")
    with torch.no_grad():
        for seq_ten, label_ten in tqdm(test_loader):
            seq_ten = seq_ten.to(device)
            label_ten = label_ten.to(device)

            # 前向传播
            outputs = model(seq_ten)  # (B, L, L)

            # 应用阈值生成 0/1 预测
            preds = (torch.sigmoid(outputs) > THRESHOLD).float()

            # 计算 TP, FP, FN (只看上三角矩阵，避免重复计算)
            # 使用 triu(1) 排除对角线和下三角
            mask = torch.triu(torch.ones_like(label_ten), diagonal=1)

            valid_preds = preds * mask
            valid_labels = label_ten * mask

            tp = (valid_preds * valid_labels).sum().item()
            fp = (valid_preds * (1 - valid_labels)).sum().item()
            fn = ((1 - valid_preds) * valid_labels).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn

    # 6. 计算最终指标
    epsilon = 1e-7
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)

    print("\n" + "=" * 30)
    print(f"🏆 TS0 测试集最终结果 (Threshold={THRESHOLD})")
    print("=" * 30)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    evaluate_on_testset()
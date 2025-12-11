import torch
import numpy as np
from torch.utils.data import DataLoader
from src.dataset import MultiFileDataset
from src.model import SpotRNAWithTransformer  # 确保这里是你最新的架构类名
from src.config import Config
from tqdm import tqdm


def greedy_decoding(prob_map, threshold=0.3):
    """
    贪心解码：确保每个碱基只配对一次
    prob_map: (L, L) numpy array
    """
    seq_len = prob_map.shape[0]
    structure = np.zeros((seq_len, seq_len))
    visited = set()

    # 1. 获取所有大于阈值的候选点
    # indices 是 (row_idx, col_idx) 的元组列表
    # prob_map 是对称的，我们只取上三角 (i < j)
    candidates = []
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if prob_map[i, j] > threshold:
                candidates.append((prob_map[i, j], i, j))

    # 2. 按概率从大到小排序
    candidates.sort(key=lambda x: x[0], reverse=True)

    # 3. 贪心选择
    for prob, i, j in candidates:
        if i not in visited and j not in visited:
            structure[i, j] = 1
            structure[j, i] = 1  # 对称
            visited.add(i)
            visited.add(j)

    return structure


def evaluate_with_postprocessing(model, dataloader, device):
    model.eval()
    total_f1 = 0
    total_p = 0
    total_r = 0
    count = 0

    print("开始评估（带后处理）...")
    with torch.no_grad():
        for seq, label in tqdm(dataloader):
            seq = seq.to(device)
            # label shape: (Batch, L, L)

            # 预测
            preds = model(seq)  # (Batch, L, L)
            probs = torch.sigmoid(preds)

            probs_np = probs.cpu().numpy()
            labels_np = label.cpu().numpy()

            for k in range(probs_np.shape[0]):
                # 单个样本处理
                prob_map = probs_np[k]
                true_map = labels_np[k]

                # === 核心：应用后处理 ===
                # threshold 设为 0.3 可以召回更多潜在配对
                pred_map = greedy_decoding(prob_map, threshold=0.3)

                # 计算 F1
                tp = np.sum(pred_map * true_map)
                fp = np.sum(pred_map) - tp
                fn = np.sum(true_map) - tp

                p = tp / (tp + fp + 1e-10)
                r = tp / (tp + fn + 1e-10)
                f1 = 2 * p * r / (p + r + 1e-10)

                total_p += p
                total_r += r
                total_f1 += f1
                count += 1

    print(f"评估完成，样本数: {count}")
    print(f"平均 Precision: {total_p / count:.4f}")
    print(f"平均 Recall:    {total_r / count:.4f}")
    print(f"平均 F1 Score:  {total_f1 / count:.4f}")


if __name__ == "__main__":
    # 1. 加载配置和模型
    # 注意：确保这里加载的是你最新的、表现最好的权重
    # 比如 Epoch 4 或 Epoch 3 的权重（虽然 F1 看起来是 0.35，但潜力最大）
    weight_path = r"D:\PycharmProjects\tRNATransferPrediction\checkpoints\model_lstm_epoch_11.pth"  # 改成你实际最新的权重路径

    model = SpotRNAWithTransformer(Config).to(Config.DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=Config.DEVICE))

    # 2. 加载数据 (只加载验证集或测试集)
    # 假设这里简单加载整个 TR0 做测试，或者你可以单独做一个 validation dataset
    dataset = MultiFileDataset(Config.DATA_DIR, max_len=Config.MAX_LEN)
    # 取一部分数据快速验证，比如前 500 个
    subset_dataset = torch.utils.data.Subset(dataset, range(500))
    dataloader = DataLoader(subset_dataset, batch_size=1, shuffle=False)

    # 3. 运行评估
    evaluate_with_postprocessing(model, dataloader, Config.DEVICE)
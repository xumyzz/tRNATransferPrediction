import torch
import numpy as np
from torch.utils.data import DataLoader
from src.dataset import MultiFileDataset, collate_pad  # 确保导入 collate_pad
from src.model import SpotRNA_LSTM_Refined
from src.config import Config
from tqdm import tqdm
import os


def greedy_decoding(prob_map, threshold=0.3):
    """
    贪心解码：确保每个碱基只配对一次
    prob_map: (L, L) numpy array, 值在 0-1 之间
    """
    # 这里的 seq_len 是包含了 padding 的长度，但我们只关心有效区域
    # 不过 prob_map 应该是已经被截取过的有效区域，或者我们在外面截取
    seq_len = prob_map.shape[0]
    structure = np.zeros((seq_len, seq_len))
    visited = set()

    # 1. 获取所有大于阈值的候选点 (只取上三角 i < j)
    candidates = []

    # 优化：使用 numpy 快速筛选，避免双重循环慢速
    # rows, cols = np.where(np.triu(prob_map, k=1) > threshold)
    # values = prob_map[rows, cols]
    # candidates = sorted(zip(values, rows, cols), key=lambda x: x[0], reverse=True)

    # 为了逻辑清晰保持循环写法，但在Python中可能稍慢
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
        # 注意：这里 dataloader 返回三个值
        for seqs, labels, masks in tqdm(dataloader):
            seqs = seqs.to(device)
            masks = masks.to(device)
            # labels 不需要去 device，因为我们在 CPU 上做评估计算

            # 预测
            # 记得传入 mask
            logits = model(seqs, mask=masks)  # (Batch, L, L)
            probs_batch = torch.sigmoid(logits)

            probs_np = probs_batch.cpu().numpy()
            labels_np = labels.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for k in range(probs_np.shape[0]):
                # 获取当前样本的有效长度
                # mask 是 (L,) 的 1/0 数组，求和即为真实长度
                valid_len = int(masks_np[k].sum())

                # 截取有效区域 (去除 Padding)
                # 这一步非常重要！否则 Padding 区域的 0 也会参与计算
                prob_map = probs_np[k, :valid_len, :valid_len]
                true_map = labels_np[k, :valid_len, :valid_len]

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

    print(f"\n评估完成，样本数: {count}")
    print(f"平均 Precision: {total_p / count:.4f}")
    print(f"平均 Recall:    {total_r / count:.4f}")
    print(f"平均 F1 Score:  {total_f1 / count:.4f}")


if __name__ == "__main__":
    # 1. 自动寻找最新的模型文件
    if not os.path.exists(Config.MODEL_SAVE_DIR):
        print(f"目录不存在: {Config.MODEL_SAVE_DIR}")
        exit()

    files = [f for f in os.listdir(Config.MODEL_SAVE_DIR) if f.endswith('.pth')]
    if not files:
        print("没有找到模型文件")
        exit()

    # 按修改时间排序找最新的，或者按名字找 epoch 最大的
    # 这里假设按 epoch 名字排序
    # files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0)
    # latest_model = files[-1]
    # weight_path = os.path.join(Config.MODEL_SAVE_DIR, latest_model)
    weight_path=r'D:\PycharmProjects\tRNATransferPrediction\checkpoints\model_transformer_epoch_9.pth'
    print(f"加载模型: {weight_path}")

    model = SpotRNA_LSTM_Refined(Config).to(Config.DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=Config.DEVICE))

    # 2. 加载数据
    dataset = MultiFileDataset(Config.DATA_DIR, max_len=Config.MAX_LEN)

    # 随机取 100 个做快速测试，或者用 full_ds
    # subset_indices = torch.randperm(len(dataset))[:100]
    # subset_dataset = torch.utils.data.Subset(dataset, subset_indices)

    # 记得使用 collate_pad 处理 batch
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_pad)

    # 3. 运行评估
    evaluate_with_postprocessing(model, dataloader, Config.DEVICE)
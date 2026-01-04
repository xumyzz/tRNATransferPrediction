import torch
import numpy as np
from torch.utils.data import DataLoader
from src.dataset import MultiFileDataset, collate_pad
from src.model import SpotRNA_LSTM_Refined  # 记得这其实是你的 LSTM+Refine 模型
from src.config import Config
from tqdm import tqdm


def greedy_decoding(prob_map, threshold=0.2):
    seq_len = prob_map.shape[0]
    structure = np.zeros((seq_len, seq_len))
    visited = set()
    candidates = []
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if prob_map[i, j] > threshold:
                candidates.append((prob_map[i, j], i, j))
    candidates.sort(key=lambda x: x[0], reverse=True)
    for prob, i, j in candidates:
        if i not in visited and j not in visited:
            structure[i, j] = 1
            structure[j, i] = 1
            visited.add(i)
            visited.add(j)
    return structure


def evaluate_ensemble(models, dataloader, device):
    for m in models:
        m.eval()

    total_f1 = 0
    total_p = 0
    total_r = 0
    count = 0

    print(f"开始集成评估 (模型数量: {len(models)})...")

    with torch.no_grad():
        for seqs, labels, masks in tqdm(dataloader):
            seqs = seqs.to(device)
            # labels 不上 GPU 节省显存

            # --- 集成核心: 累加所有模型的输出 ---
            avg_probs = None
            for model in models:
                # 记得传入 mask，虽然 LSTM 可能不用，但保持接口一致
                logits = model(seqs, mask=masks.to(device))
                probs = torch.sigmoid(logits)

                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs += probs

            # 取平均
            avg_probs /= len(models)

            # 转 numpy
            probs_np = avg_probs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for k in range(probs_np.shape[0]):
                valid_len = int(masks_np[k].sum())
                prob_map = probs_np[k, :valid_len, :valid_len]
                true_map = labels_np[k, :valid_len, :valid_len]

                # 后处理
                pred_map = greedy_decoding(prob_map, threshold=0.4)

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

    print(f"集成评估完成，样本数: {count}")
    print(f"平均 Precision: {total_p / count:.4f}")
    print(f"平均 Recall:    {total_r / count:.4f}")
    print(f"平均 F1 Score:  {total_f1 / count:.4f}")


if __name__ == "__main__":
    # 1. 定义模型列表
    models = []

    # 这里填入你 Epoch 9 和 Epoch 10 (甚至 Epoch 4) 的权重路径
    # 假设你保存的文件名包含 epoch 信息
    checkpoint_paths = [
        r"D:\PycharmProjects\tRNATransferPrediction\checkpoints\model_transformer_epoch_1.pth",
        r"D:\PycharmProjects\tRNATransferPrediction\checkpoints\model_transformer_epoch_2.pth",
        r"D:\PycharmProjects\tRNATransferPrediction\checkpoints\model_transformer_epoch_3.pth"
    ]

    print("加载模型中...")
    for path in checkpoint_paths:
        m = SpotRNA_LSTM_Refined(Config).to(Config.DEVICE)
        m.load_state_dict(torch.load(path, map_location=Config.DEVICE))
        models.append(m)

    # 2. 数据集
    dataset = MultiFileDataset(Config.DATA_DIR, max_len=Config.MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE * 2, shuffle=False, collate_fn=collate_pad)

    # 3. 跑起来！
    evaluate_ensemble(models, dataloader, Config.DEVICE)
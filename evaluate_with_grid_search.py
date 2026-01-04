import torch
import numpy as np
from torch.utils.data import DataLoader
from src.dataset import MultiFileDataset, collate_pad
from src.model import SpotRNA_LSTM_Refined  # 确保这里引用的类名和你训练时一致
from src.config import Config
from tqdm import tqdm


# --- 贪心解码函数 (带阈值) ---
def greedy_decoding(prob_map, threshold=0.3):
    seq_len = prob_map.shape[0]
    structure = np.zeros((seq_len, seq_len))
    visited = set()
    candidates = []

    # 1. 筛选大于阈值的点
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if prob_map[i, j] > threshold:
                candidates.append((prob_map[i, j], i, j))

    # 2. 按概率从大到小排序
    candidates.sort(key=lambda x: x[0], reverse=True)

    # 3. 贪心选择（互斥）
    for prob, i, j in candidates:
        if i not in visited and j not in visited:
            structure[i, j] = 1
            structure[j, i] = 1
            visited.add(i)
            visited.add(j)

    return structure


# --- 核心：网格搜索评估函数 ---
def evaluate_with_grid_search(models, dataloader, device):
    for m in models: m.eval()

    # 这里设定我们要扫描的阈值范围
    # 根据你的趋势，重点扫描 0.25 - 0.35 区域
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40]

    # 初始化统计器：记录每个阈值的总 P, R, F1
    # 这里的 F1 采用 Macro-Average (先算每个样本的F1，再求平均)，和你之前的指标一致
    metrics = {t: {'f1_sum': 0, 'p_sum': 0, 'r_sum': 0} for t in thresholds}
    count = 0

    print(f"🚀 开始网格搜索，测试阈值: {thresholds} ...")

    with torch.no_grad():
        for seqs, labels, masks in tqdm(dataloader):
            seqs = seqs.to(device)
            # labels 不上 GPU 节省显存

            # 1. 模型集成推理 (最耗时，只做一次)
            avg_probs = None
            for model in models:
                logits = model(seqs, mask=masks.to(device))
                probs = torch.sigmoid(logits)
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs += probs
            avg_probs /= len(models)

            probs_np = avg_probs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            masks_np = masks.cpu().numpy()

            # 2. 针对不同阈值循环解码 (纯 CPU 计算，很快)
            batch_size = probs_np.shape[0]
            for k in range(batch_size):
                valid_len = int(masks_np[k].sum())
                prob_map = probs_np[k, :valid_len, :valid_len]
                true_map = labels_np[k, :valid_len, :valid_len]

                for t in thresholds:
                    # 使用当前阈值解码
                    pred_map = greedy_decoding(prob_map, threshold=t)

                    # 计算指标
                    tp = np.sum(pred_map * true_map)
                    fp = np.sum(pred_map) - tp
                    fn = np.sum(true_map) - tp

                    p = tp / (tp + fp + 1e-10)
                    r = tp / (tp + fn + 1e-10)
                    f1 = 2 * p * r / (p + r + 1e-10)

                    metrics[t]['p_sum'] += p
                    metrics[t]['r_sum'] += r
                    metrics[t]['f1_sum'] += f1

                count += 1

    print(f"\n📊 === 最终结果报告 (样本数: {count}) ===")

    best_avg_f1 = 0
    best_t = 0

    # 打印表头
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 50)

    for t in thresholds:
        avg_p = metrics[t]['p_sum'] / count
        avg_r = metrics[t]['r_sum'] / count
        avg_f1 = metrics[t]['f1_sum'] / count

        print(f"{t:<10.2f} | {avg_p:<10.4f} | {avg_r:<10.4f} | {avg_f1:<10.4f}")

        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_t = t

    print("-" * 50)
    print(f"🏆 最佳阈值: {best_t} | 最佳 F1: {best_avg_f1:.4f}")


# --- 主函数入口 ---
if __name__ == "__main__":
    # 1. 定义模型列表 (填入你微调后的 Epoch 1, 2, 3 权重)
    # 请务必修改这里的路径！！！
    checkpoint_paths = [
        r"D:\PycharmProjects\tRNATransferPrediction\checkpoints\model_transformer_epoch_1.pth",  # 填你的文件名
        r"D:\PycharmProjects\tRNATransferPrediction\checkpoints\model_transformer_epoch_2.pth",
        r"D:\PycharmProjects\tRNATransferPrediction\checkpoints\model_transformer_epoch_3.pth"
    ]

    print(f"正在加载 {len(checkpoint_paths)} 个模型...")
    models = []
    for path in checkpoint_paths:
        try:
            m = SpotRNA_LSTM_Refined(Config).to(Config.DEVICE)
            # 注意 map_location，防止显存不足
            m.load_state_dict(torch.load(path, map_location=Config.DEVICE))
            models.append(m)
            print(f"成功加载: {path}")
        except Exception as e:
            print(f"加载失败 {path}: {e}")

    if not models:
        print("没有加载到任何模型，请检查路径！")
        exit()

    # 2. 准备数据集
    print("正在加载数据集...")
    dataset = MultiFileDataset(Config.DATA_DIR, max_len=Config.MAX_LEN)
    # batch_size 稍微大点跑得快，只要不爆显存
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE * 4, shuffle=False, collate_fn=collate_pad)

    # 3. 运行网格搜索
    evaluate_with_grid_search(models, dataloader, Config.DEVICE)
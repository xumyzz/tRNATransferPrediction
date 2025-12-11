import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 从 src 包中导入必要的模块 ---
# 确保你的 PyCharm 已经把 src 标记为 Source Root，或者直接在根目录运行此脚本
from src.config import Config
from src.model import SpotRNAWithLSTM
from src.dataset import MultiFileDataset


def plot_rna_comparison(model, dataset, device, num_samples=3, threshold=0.5, save_dir="vis_results"):
    """
    随机抽取样本并画图对比
    :param model: 已加载权重的模型
    :param dataset: 验证集/测试集
    :param num_samples: 要画几张图
    :param threshold: 二值化阈值 (默认 0.5)
    :param save_dir: 图片保存文件夹
    """
    model.eval()

    # 创建保存文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 随机选择索引
    if len(dataset) < num_samples:
        num_samples = len(dataset)
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    print(f"正在生成 {num_samples} 张可视化对比图...")

    for idx in indices:
        # 获取单个样本
        # 注意：dataset[idx] 返回的是 (seq_tensor, label_matrix)
        seq_ten, label_mat = dataset[idx]

        # 增加 Batch 维度以便输入模型: (L, 4) -> (1, L, 4)
        seq_input = seq_ten.unsqueeze(0).to(device)

        # 预测
        with torch.no_grad():
            logits = model(seq_input)
            # Sigmoid 归一化到 0~1
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # 结果是 (L, L)

        # 获取真实标签 (转为 numpy)
        true_mat = label_mat.numpy()  # (L, L)

        # 二值化预测 (大于阈值算配对)
        pred_binary = (probs > threshold).astype(float)

        # --- 开始画图 (三联图) ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        L = true_mat.shape[0]  # 序列长度

        # 1. 左图: 真实标签 (Ground Truth)
        # 使用 Greys 配色: 白色背景，黑色点
        axes[0].imshow(true_mat, cmap='Greys', interpolation='nearest')
        axes[0].set_title(f"Ground Truth (Len={L})")
        axes[0].set_xlabel("Base Index")
        axes[0].set_ylabel("Base Index")

        # 2. 中图: 预测概率 (Predicted Probability)
        # 使用 hot_r 配色: 白色是0，红/黄是高概率
        im = axes[1].imshow(probs, cmap='hot_r', vmin=0, vmax=1, interpolation='nearest')
        axes[1].set_title("Predicted Probability Heatmap")
        # 添加颜色条
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # 3. 右图: 二值化预测 (Binary Prediction)
        axes[2].imshow(pred_binary, cmap='Greys', interpolation='nearest')
        axes[2].set_title(f"Binary Prediction (Thresh={threshold})")

        plt.tight_layout()

        # 保存图片
        save_path = os.path.join(save_dir, f"sample_{idx}_len{L}.png")
        plt.savefig(save_path, dpi=150)
        print(f"--> 已保存: {save_path}")

        # 如果在 PyCharm SciView 或者 Jupyter 里想看，可以把下面这行注释打开
        # plt.show()

        plt.close()  # 关闭画板释放内存


def main():
    print(f"使用设备: {Config.DEVICE}")

    # 1. 指定要可视化的权重文件路径
    # === 修改这里 ===
    # 比如使用你刚才训练好的 Epoch 10 的权重
    # 如果权重在 src/checkpoints 下，路径可能是 "src/checkpoints/model_lstm_epoch_10.pth"
    # 或者直接用 Config.PRETRAINED_PATH (如果你刚才改过 Config)
    checkpoint_path = r"D:\PycharmProjects\tRNATransferPrediction\Baseline\baseline_best_f1_0.61.pth"
    # ==============

    if not os.path.exists(checkpoint_path):
        print(f"错误：找不到权重文件 {checkpoint_path}")
        print("请修改脚本中的 checkpoint_path 变量为正确路径。")
        return

    # 2. 准备数据 (只读几个用于测试)
    # 必须保持和训练时一样的 MaxLen，否则卷积层尺寸可能对不上
    dataset = MultiFileDataset(Config.DATA_DIR, max_len=Config.MAX_LEN)

    if len(dataset) == 0:
        print("错误：未找到数据，请检查 Config.DATA_DIR")
        return

    # 3. 初始化模型结构
    # 必须和训练时的参数完全一致
    model = SpotRNAWithLSTM(
        num_resnet_layers=Config.RESNET_LAYERS,
        hidden_dim=Config.HIDDEN_DIM,
        lstm_hidden=Config.LSTM_HIDDEN
    ).to(Config.DEVICE)

    # 4. 加载权重
    try:
        # 加上 weights_only=True 消除警告 (如果报错把这个参数删掉)
        state_dict = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"成功加载权重: {checkpoint_path}")
    except Exception as e:
        # 兼容旧版本 PyTorch
        print(f"带参数加载失败，尝试默认加载... ({e})")
        state_dict = torch.load(checkpoint_path, map_location=Config.DEVICE)
        model.load_state_dict(state_dict)
        print(f"成功加载权重: {checkpoint_path}")

    # 5. 运行绘图函数
    # num_samples: 想画几张图
    # threshold: 判定为配对的概率阈值
    plot_rna_comparison(model, dataset, Config.DEVICE, num_samples=5, threshold=0.5)
    print("\n全部完成！请查看 vis_results 文件夹。")


if __name__ == "__main__":
    main()
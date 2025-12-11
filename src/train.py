import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os

# --- 导入我们拆分好的模块 ---
from .config import Config  # 导入配置
from .utils import compute_masked_loss, calculate_f1  # 导入工具函数
from .dataset import MultiFileDataset, collate_pad  # 假设你已经有了这个文件
from .model import SpotRNAWithLSTM  # 假设你已经有了这个文件
from .model import SpotRNAWithTransformer

def train():
    print(f"使用设备: {Config.DEVICE}")

    # --- 1. 准备数据 ---
    # 直接使用 Config 中的参数
    full_ds = MultiFileDataset(Config.DATA_DIR, max_len=Config.MAX_LEN)

    if len(full_ds) == 0:
        print("错误：没有数据，请检查路径。")
        return

    # 划分验证集
    train_len = int(0.9 * len(full_ds))
    val_len = len(full_ds) - train_len
    # 使用 random_split
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_pad)

    # --- 2. 初始化模型 --- 此处使用ResNet+LSTM
    # model = SpotRNAWithLSTM(
    #     num_resnet_layers=Config.RESNET_LAYERS,
    #     hidden_dim=Config.HIDDEN_DIM,
    #     lstm_hidden=Config.LSTM_HIDDEN
    # ).to(Config.DEVICE)

    #此处使用ResNet+Transformer
    model = SpotRNAWithTransformer(
        Config
    ).to(Config.DEVICE)

    # ====== 【新增】加载预训练权重 ======
    if Config.PRETRAINED_PATH and os.path.exists(Config.PRETRAINED_PATH):
        print(f"正在加载预训练权重: {Config.PRETRAINED_PATH}")
        try:
            # 加载权重
            state_dict = torch.load(Config.PRETRAINED_PATH, map_location=Config.DEVICE)
            model.load_state_dict(state_dict)
            print(">>> 权重加载成功！将在现有基础上继续训练。")
        except Exception as e:
            print(f"!!! 权重加载失败: {e}")
            return # 或者选择继续从头训练
    else:
        print("未指定预训练权重，将从头开始训练。")
    # ====================================

    optimizer = optim.Adam(model.parameters(), lr=Config.LR,weight_decay=Config.WEIGHT_DECAY)

    # --- 3. Loss 定义 ---
    pos_weight_tensor = torch.tensor([Config.POS_WEIGHT]).to(Config.DEVICE)
    criterion_raw = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction='none')

    # --- 4. 开始训练循环 ---
    print(f"\n开始训练 (Epochs={Config.EPOCHS}, Accum={Config.ACCUM_STEPS})...")

    for epoch in range(Config.EPOCHS):
        model.train()
        optimizer.zero_grad()  # 清理上一轮残留
        total_loss = 0

        for batch_idx, (seqs, labels, masks) in enumerate(train_loader):
            seqs = seqs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)  # 别忘了把 labels 也放进去
            masks = masks.to(Config.DEVICE)  # masks 也要放进去

            logits = model(seqs)

            # 调用 utils.py 中的函数计算 loss
            loss = compute_masked_loss(logits, labels, masks, criterion_raw)

            # 梯度累积
            loss = loss / Config.ACCUM_STEPS
            loss.backward()

            # 记录还原后的真实 Loss
            current_real_loss = loss.item() * Config.ACCUM_STEPS
            total_loss += current_real_loss

            # 达到累积步数进行更新
            if (batch_idx + 1) % Config.ACCUM_STEPS == 0:
                # 可选：梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                optimizer.step()
                optimizer.zero_grad()

            if batch_idx % 50 == 0:
                print(f"Step [{batch_idx}] Loss: {current_real_loss:.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"=== Epoch {epoch + 1} 结束, 平均 Loss: {avg_loss:.4f} ===")

        # --- 5. 验证 (Validation) ---
        model.eval()
        val_loss = 0
        val_f1 = 0

        with torch.no_grad():
            for seqs, labels, masks in val_loader:
                seqs = seqs.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                masks = masks.to(Config.DEVICE)

                logits = model(seqs)

                # 使用同样的工具函数计算验证 Loss
                loss = compute_masked_loss(logits, labels, masks, criterion_raw)
                val_loss += loss.item()

                # 使用工具函数计算 F1
                f1 = calculate_f1(logits, labels, masks)
                val_f1 += f1

        print(f"=== 验证集 Loss: {val_loss / len(val_loader):.4f} | F1: {val_f1 / len(val_loader):.4f} ===\n")

        # 保存模型
        save_path = os.path.join(Config.MODEL_SAVE_DIR, f"model_lstm_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    train()
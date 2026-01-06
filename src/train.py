import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import os

# --- 导入我们拆分好的模块 ---
from .config import Config  # 导入配置
from .utils import compute_masked_loss, calculate_f1  # 导入工具函数
from .dataset import MultiFileDataset, collate_pad,MultiFileDatasetUpgrade  # 假设你已经有了这个文件
# from .model import SpotRNAWithLSTM  # 假设你已经有了这个文件
from .model import SpotRNA_LSTM_Refined
from .cluster_split import parse_cdhit_clstr, create_cluster_splits, save_split_config

def train(clstr_path=None, split_seed=42, train_frac=0.8, val_frac=0.1, split_out=None):
    print(f"使用设备: {Config.DEVICE}")

    # --- 1. 准备数据 ---
    # 直接使用 Config 中的参数
    full_ds = MultiFileDatasetUpgrade(Config.DATA_DIR, max_len=Config.MAX_LEN)

    if len(full_ds) == 0:
        print("错误：没有数据，请检查路径。")
        return

    # 划分验证集 - 使用聚类分割或随机分割
    if clstr_path and os.path.exists(clstr_path):
        print(f"\n🔬 使用聚类文件进行无泄漏分割: {clstr_path}")
        
        # Parse cluster file
        name_to_cluster = parse_cdhit_clstr(clstr_path)
        
        # Create cluster-based splits
        train_indices, val_indices, test_indices = create_cluster_splits(
            full_ds, 
            name_to_cluster,
            train_frac=train_frac,
            val_frac=val_frac,
            seed=split_seed
        )
        
        # Create Subset datasets
        train_ds = Subset(full_ds, train_indices)
        val_ds = Subset(full_ds, val_indices)
        
        # Save split configuration if requested
        if split_out:
            metadata = {
                "clstr_path": clstr_path,
                "split_seed": split_seed,
                "train_frac": train_frac,
                "val_frac": val_frac,
                "n_clusters": len(set(name_to_cluster.values())),
                "max_len": Config.MAX_LEN
            }
            save_split_config(split_out, train_indices, val_indices, test_indices, metadata)
    else:
        print("\n🎲 使用随机分割 (未指定聚类文件)")
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
    model = SpotRNA_LSTM_Refined(
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
    # criterion_raw = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction='none')

    # --- 4. 开始训练循环 ---
    print(f"\n开始训练 (Epochs={Config.EPOCHS}, Accum={Config.ACCUM_STEPS})...")

    for epoch in range(Config.EPOCHS):
        model.train()
        optimizer.zero_grad()  # 清理上一轮残留
        total_loss = 0

        for batch_idx, (seqs, labels, masks) in enumerate(train_loader):
            seqs = seqs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)

            # === 修改 1: 传入 mask ===
            logits = model(seqs, mask=masks)

            # === 修改 2: 调用 utils 计算 loss (带 pos_weight) ===
            # 直接在这里传入 pos_weight，不依赖外部 criterion
            loss = compute_masked_loss(logits, labels, masks, pos_weight=Config.POS_WEIGHT)

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

                # [修正1] 验证时也要传入 mask，避免 Padding 干扰
                logits = model(seqs, mask=masks)

                # [修正2] 这里不能传 criterion_raw 了，要传 pos_weight
                loss = compute_masked_loss(logits, labels, masks, pos_weight=Config.POS_WEIGHT)

                val_loss += loss.item()

                # 使用工具函数计算 F1
                f1 = calculate_f1(logits, labels, masks)
                val_f1 += f1

        print(f"=== 验证集 Loss: {val_loss / len(val_loader):.4f} | F1: {val_f1 / len(val_loader):.4f} ===\n")

        # 保存模型 (建议文件名改得更有意义一点)
        save_path = os.path.join(Config.MODEL_SAVE_DIR, f"model_transformer_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    train()
import torch


class Config:
    # 路径设置
    DATA_DIR = r"D:\PycharmProjects\tRNATransferPrediction\data\TR0"  # 假设数据在 data 文件夹
    MODEL_SAVE_DIR = r"D:\PycharmProjects\tRNATransferPrediction\checkpoints"

    # 数据参数
    MAX_LEN = 300

    # 训练参数
    BATCH_SIZE = 2
    ACCUM_STEPS = 32
    EPOCHS = 10
    LR = 0.0001
    POS_WEIGHT = 3.5
    WEIGHT_DECAY = 1e-4
    # 模型参数
    RESNET_LAYERS = 8
    HIDDEN_DIM = 64
    LSTM_HIDDEN = 64
    PRETRAINED_PATH = r'D:\PycharmProjects\tRNATransferPrediction\checkpoints\model_transformer_epoch_4.pth'

    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

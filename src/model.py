import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResNetBlock1D(nn.Module):
    """1D ResNet Block for Sequence Features"""

    def __init__(self, channels, kernel_size=3, dilation=1):
        super(ResNetBlock1D, self).__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        residual = x
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.elu(self.bn2(self.conv2(out)))
        out += residual
        return out


class ResNetBlock2D(nn.Module):
    """2D ResNet Block for Structure Refinement (The Missing Piece)"""

    def __init__(self, channels):
        super(ResNetBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        residual = x
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.elu(self.bn2(self.conv2(out)))
        out += residual
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SpotRNA_LSTM_Refined(nn.Module):
    # 注意：为了兼容你现有的调用，我不改类名，但实质内容已经换成了 LSTM
    # 建议之后你在项目中把它重命名为 SpotRNA_LSTM_Refined
    def __init__(self, config):
        super(SpotRNA_LSTM_Refined, self).__init__()

        self.hidden_dim = config.HIDDEN_DIM  # 建议 64
        self.num_res1d = config.RESNET_LAYERS  # 建议 8-10
        # 假设 config 中有 LSTM_HIDDEN，如果没有，默认用 hidden_dim
        self.lstm_hidden = getattr(config, 'LSTM_HIDDEN', self.hidden_dim)

        # --- Stage 1: Sequence Embedding & 1D CNN ---
        self.embedding = nn.Linear(4, self.hidden_dim)

        # 1D Local Features (ResNet)
        self.resnet1d_layers = nn.ModuleList([
            ResNetBlock1D(self.hidden_dim, dilation=2 ** min(i, 4))
            for i in range(self.num_res1d)
        ])

        # --- Stage 2: Sequence Context (Replaced Transformer with BiLSTM) ---
        # LSTM 能够更稳定地处理 tRNA 这种长度的序列依赖
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.lstm_hidden,
            num_layers=1,  # 1层 BiLSTM 通常足够
            batch_first=True,
            bidirectional=True
        )

        # --- Stage 3: 1D to 2D Projection ---
        # 拼接策略：
        # Input to 2D = [ResNet_Feature(Local) + LSTM_Feature(Context)]
        # 维度计算:
        #   ResNet: hidden_dim
        #   LSTM:   lstm_hidden * 2 (双向)
        #   Outer Concat (i, j): (hidden + 2*lstm) * 2

        dim_1d = self.hidden_dim + self.lstm_hidden * 2
        dim_2d_input = dim_1d * 2

        # 降维层：把拼接后巨大的维度降下来，方便跑 ResNet2D
        # 比如从 300+ 降到 64
        self.proj_2d = nn.Sequential(
            nn.Conv2d(dim_2d_input, self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ELU()
        )

        # --- Stage 4: 2D Refinement (Retained & Critical) ---
        # 这是提分的关键，用于修补接触图
        self.resnet2d_layers = nn.Sequential(
            *[ResNetBlock2D(self.hidden_dim) for _ in range(5)]
        )

        # Output
        self.final_conv = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)

    def forward(self, x, mask=None):
        # x: (B, L, 4)
        B, L, _ = x.shape

        # 1. Embed & 1D ResNet (Local)
        x_emb = self.embedding(x)
        x_local = x_emb.permute(0, 2, 1)
        for layer in self.resnet1d_layers:
            x_local = layer(x_local)
        x_local = x_local.permute(0, 2, 1)  # (B, L, hidden)

        # 2. BiLSTM (Global Context)
        # 处理 PackSequence 是最佳实践，但为了代码简洁且 tRNA 长度差异不大，直接跑也可以
        # 如果追求极致，这里可以用 pack_padded_sequence
        x_lstm, _ = self.lstm(x_local)  # (B, L, 2*lstm_hidden)

        # Combine Local + Global -> (B, L, hidden + 2*lstm_hidden)
        x_1d = torch.cat([x_local, x_lstm], dim=-1)

        # 3. Outer Concatenation (1D -> 2D)
        # 广播拼接 (i, j)
        x_row = x_1d.unsqueeze(2).expand(-1, -1, L, -1)
        x_col = x_1d.unsqueeze(1).expand(-1, L, -1, -1)
        x_2d = torch.cat([x_row, x_col], dim=-1)  # (B, L, L, dim_2d_input)
        x_2d = x_2d.permute(0, 3, 1, 2)  # (B, C, L, L)

        # 4. Project & Refine (CNN 修图)
        x_2d = self.proj_2d(x_2d)
        x_2d = self.resnet2d_layers(x_2d)

        # 5. Output
        logits = self.final_conv(x_2d).squeeze(1)  # (B, L, L)

        # Symmetrize (保证输出矩阵对称)
        logits = (logits + logits.transpose(1, 2)) / 2
        return logits


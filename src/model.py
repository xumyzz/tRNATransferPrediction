import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResNetBlock(nn.Module):
    """
    简单的残差块 (ResNet Block)
    对应图中 Block A 的简化版
    """

    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        # 保持输入输出维度一致，padding设置为same
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        residual = x

        # 第一层卷积
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.dropout(out)

        # 第二层卷积
        out = F.elu(self.bn2(self.conv2(out)))

        # 残差连接 (Skip Connection): 输入直接加到输出上
        out += residual
        return out


class SpotRNAWithLSTM(nn.Module):
    def __init__(self, num_resnet_layers=10, hidden_dim=32, lstm_hidden=32):
        super(SpotRNAWithLSTM, self).__init__()

        # 1. 输入转换层
        # 输入是 RNA 序列的 One-hot (4个通道: A,U,C,G)
        # 经过 Outer Concatenation 变成 8 个通道
        self.input_channels = 8

        # 初始卷积层 (对应图中的 Initial 3x3 convolution)
        self.initial_conv = nn.Conv2d(self.input_channels, hidden_dim, kernel_size=3, padding=1)

        # 2. 堆叠 ResNet Block (对应图中的 Block A x NA)
        self.resnet_blocks = nn.Sequential(
            *[ResNetBlock(hidden_dim) for _ in range(num_resnet_layers)]
        )
        self.lstm_row = nn.LSTM(input_size=hidden_dim, hidden_size=lstm_hidden, batch_first=True, bidirectional=True)
        self.lstm_col = nn.LSTM(input_size=hidden_dim, hidden_size=lstm_hidden, batch_first=True, bidirectional=True)
        final_in_dim = hidden_dim + (lstm_hidden * 2) * 2

        # 3. 输出层 (对应图中的 Output layer)
        # 将特征图压缩回 1 个通道 (代表配对概率)
        self.final_conv = nn.Conv2d(final_in_dim, 1, kernel_size=1)

    def forward(self, seq_onehot):
        """
        seq_onehot shape: (Batch, Length, 4)
        """
        B, L, _ = seq_onehot.shape

        # --- 步骤 A: Outer Concatenation (构建 2D 特征图) ---
        # 这一步将 1D 序列变成 2D 矩阵。
        # 比如 i 位置是 A，j 位置是 U，那么 (i, j) 点就有 A和U 的特征。

        # (Batch, L, 1, 4) -> 扩充维度
        x_row = seq_onehot.unsqueeze(2).expand(-1, -1, L, -1)
        # (Batch, 1, L, 4) -> 扩充维度
        x_col = seq_onehot.unsqueeze(1).expand(-1, L, -1, -1)

        # 拼接: (Batch, L, L, 8)
        x_2d = torch.cat([x_row, x_col], dim=-1)

        # 调整维度以符合 PyTorch 卷积输入: (Batch, Channels=8, Height=L, Width=L)
        x_2d = x_2d.permute(0, 3, 1, 2)

        # --- 步骤 B: 深度网络处理 ---
        x_res = self.initial_conv(x_2d)
        x_res = self.resnet_blocks(x_res)

        # --- Step 3: 2D-LSTM ---
        # 准备数据：我们需要把 (B, C, L, L) 变成序列格式

        # A. Row-wise LSTM
        # Permute to (B*L, L, C) -> 把每一行当做一个独立的序列
        # B:Batch, C:Channel, H:Height(Row), W:Width(Col)
        x_for_lstm_row = x_res.permute(0, 2, 3, 1).contiguous().view(B * L, L, -1)
        out_row, _ = self.lstm_row(x_for_lstm_row)
        # out_row: (B*L, L, 2*hidden) -> 还原回 (B, L, L, 2*hidden) -> Permute (B, 2*hidden, L, L)
        out_row = out_row.view(B, L, L, -1).permute(0, 3, 1, 2)

        # B. Col-wise LSTM (同理，但先转置一下让列变成行)
        # 转置 H 和 W -> (B, C, W, H)
        x_for_lstm_col = x_res.permute(0, 1, 3, 2)
        # 变成序列 -> (B*W, H, C)
        x_for_lstm_col = x_for_lstm_col.permute(0, 2, 3, 1).contiguous().view(B * L, L, -1)
        out_col, _ = self.lstm_col(x_for_lstm_col)
        # 还原 -> (B, L, L, 2*hidden) -> Permute (B, 2*hidden, L, L) -> 转置回原位 (B, 2*hidden, L, L)
        # 注意 out_col 现在的维度对应的空间是 (W, H)，需要把最后两维转回来
        out_col = out_col.view(B, L, L, -1).permute(0, 3, 2, 1)  # 注意这里的 2,1 是为了把 H,W 转回来

        # --- Step 4: Concatenation & Output ---
        # 把 ResNet原始特征 + 行LSTM特征 + 列LSTM特征 拼起来
        # 这种 "Skip Connection" 很重要，保证了局部特征不丢失
        x_final = torch.cat([x_res, out_row, out_col], dim=1)

        logits = self.final_conv(x_final)

        logits = (logits + logits.transpose(2, 3)) / 2
        return logits.squeeze(1)






class PositionalEncoding(nn.Module):
    """
    标准的 Transformer 正弦位置编码
    """

    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为 buffer，不是参数，不更新
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim)
        return x + self.pe[:, :x.size(1), :]


class ResNetBlockForTransformer(nn.Module):
    """
    1D ResNet Block
    """

    def __init__(self, channels):
        super(ResNetBlockForTransformer, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.elu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.elu(out)
        out += residual
        return out


class SpotRNAWithTransformer(nn.Module):
    def __init__(self, config):
        super(SpotRNAWithTransformer, self).__init__()

        # === 升级 1: 增加维度 ===
        # 建议在 config 里把 HIDDEN_DIM 改成 64 或 128。
        # 这里我们硬编码倍增，保证容量足够。
        self.hidden_dim = config.HIDDEN_DIM
        self.num_layers = config.RESNET_LAYERS

        # Embedding
        self.embedding = nn.Linear(4, self.hidden_dim)

        # ResNet (提取局部特征)
        self.resnet_layers = nn.ModuleList([
            ResNetBlockForTransformer(self.hidden_dim)
            for _ in range(self.num_layers)
        ])

        # === 升级 2: 位置编码 ===
        self.pos_encoder = PositionalEncoding(self.hidden_dim, max_len=config.MAX_LEN + 50)

        # Transformer (提取全局特征)
        # 确保 nhead 能被 hidden_dim 整除
        nhead = 4
        if self.hidden_dim % nhead != 0:
            nhead = 1  # 兜底

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=nhead,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # === 升级 3: 融合分类头 ===
        # 我们把 ResNet 输出 (局部) + Transformer 输出 (全局) 拼起来
        # 所以维度是 hidden_dim * 2 (单个位置) -> 对称拼接后是 hidden_dim * 4
        self.fc_out = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, 64),  # 输入维度翻倍
            nn.ELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (Batch, Len, 4)

        # 1. Embedding
        x = self.embedding(x)  # (B, L, C)

        # 2. ResNet (局部特征)
        x_res = x.permute(0, 2, 1)  # (B, C, L)
        for layer in self.resnet_layers:
            x_res = layer(x_res)
        x_res = x_res.permute(0, 2, 1)  # (B, L, C) -> 记住这个，这是局部特征

        # 3. Transformer (全局特征)
        # 加上位置编码
        x_trans = self.pos_encoder(x_res)
        x_trans = self.transformer(x_trans)  # (B, L, C) -> 这是全局特征

        # 4. 特征融合 (Skip Connection)
        # 把局部和全局拼起来: (B, L, 2*C)
        x_combined = torch.cat([x_res, x_trans], dim=-1)

        # 5. 生成 2D 图
        B, L, C_double = x_combined.shape  # C_double = 2*C

        # 广播拼接 (i, j)
        x_i = x_combined.unsqueeze(2).expand(B, L, L, C_double)
        x_j = x_combined.unsqueeze(1).expand(B, L, L, C_double)

        # 最终特征: (B, L, L, 4*C)
        x_map = torch.cat([x_i, x_j], dim=-1)

        # 6. 分类
        logits = self.fc_out(x_map)
        logits = logits.squeeze(-1)
        logits = (logits + logits.transpose(1, 2)) / 2

        return logits
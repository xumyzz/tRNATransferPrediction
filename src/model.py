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


class MambaBlock2D(nn.Module):
    """
    针对 L x L 矩阵设计的 2D-Mamba 模块 (四向交叉扫描 Cross-Scan)
    替代笨重且感受野有限的 2D CNN
    """
    def __init__(self, d_model):
        super(MambaBlock2D, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        
        # 实例化标准 1D Mamba
        # 这里参数非常轻量，d_state=16 是 SSM 的标准状态维度
        self.mamba = Mamba(
            d_model=d_model, 
            d_state=16, 
            d_conv=4, 
            expand=2  # 扩展因子，相当于 MLP 的放大倍数，2 已经足够
        )

    def forward(self, x):
        # x shape: (B, C, H, W) 其中 H = W = L
        B, C, H, W = x.shape
        
        # Mamba 期待输入特征在最后一个维度 (B, L, D)，所以先 permute 并做归一化
        x_perm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_norm = self.norm(x_perm)
        
        # ========== 四向交叉扫描 (Cross-Scan Mechanism) ==========
        
        # 1. 按行正向扫描 (Top-Left to Bottom-Right)
        x_row_fwd = x_norm.reshape(B, H * W, C)
        out1 = self.mamba(x_row_fwd).reshape(B, H, W, C)
        
        # 2. 按行反向扫描 (Bottom-Right to Top-Left)
        x_row_rev = torch.flip(x_row_fwd, dims=[1])
        out2_rev = self.mamba(x_row_rev)
        out2 = torch.flip(out2_rev, dims=[1]).reshape(B, H, W, C)
        
        # 3. 按列正向扫描 (Transpose -> Top-Left to Bottom-Right)
        x_col_fwd = x_norm.transpose(1, 2).reshape(B, H * W, C)
        out3_col = self.mamba(x_col_fwd)
        out3 = out3_col.reshape(B, W, H, C).transpose(1, 2)
        
        # 4. 按列反向扫描 
        x_col_rev = torch.flip(x_col_fwd, dims=[1])
        out4_rev = self.mamba(x_col_rev)
        out4 = torch.flip(out4_rev, dims=[1]).reshape(B, W, H, C).transpose(1, 2)
        
        # =========================================================
        
        # 聚合四个方向的全局特征 (直接相加融合)
        out = out1 + out2 + out3 + out4
        
        # 还原回 (B, C, H, W) 的格式
        out = out.permute(0, 3, 1, 2)
        
        # 残差连接
        return x + out

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
            nn.Conv2d(dim_2d_input+3, self.hidden_dim, kernel_size=1),
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

    # ====== 新增：构建基于物理规则的多通道 2D 先验 (UFold 思想) ======
        # 提取各个碱基的 boolean mask (B, L, 1)
        A = x[:, :, 0:1]
        C = x[:, :, 1:2]
        G = x[:, :, 2:3]
        U = x[:, :, 3:4]

        # 计算配对矩阵 (B, 1, L, L)
        # 例如 A_i * U_j + U_i * A_j
        AU_pair = torch.bmm(A, U.transpose(1, 2)) + torch.bmm(U, A.transpose(1, 2))
        CG_pair = torch.bmm(C, G.transpose(1, 2)) + torch.bmm(G, C.transpose(1, 2))
        GU_pair = torch.bmm(G, U.transpose(1, 2)) + torch.bmm(U, G.transpose(1, 2))
        prior_2d = torch.stack([AU_pair, CG_pair, GU_pair], dim=1)

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

        # ====== 修改：把多通道 2D 先验拼接到网络提取的 2D 特征中 ======
        x_2d = torch.cat([x_2d, prior_2d], dim=1) # 现在维度是 dim_2d_input + 3
        # ===========================================================

        # 4. Project & Refine (CNN 修图)
        x_2d = self.proj_2d(x_2d)
        x_2d = self.resnet2d_layers(x_2d)

        # 5. Output
        logits = self.final_conv(x_2d).squeeze(1)  # (B, L, L)

        # Symmetrize (保证输出矩阵对称)
        logits = (logits + logits.transpose(1, 2)) / 2
        return logits

class TRNA_Mamba_Refined(nn.Module):
    def __init__(self, config):
        super(TRNA_Mamba_Refined, self).__init__()

        self.hidden_dim = config.HIDDEN_DIM  # 建议 64
        self.num_res1d = config.RESNET_LAYERS  # 建议 8-10
        self.lstm_hidden = getattr(config, 'LSTM_HIDDEN', self.hidden_dim)

        # --- Stage 1: Sequence Embedding & 1D CNN ---
        self.embedding = nn.Linear(4, self.hidden_dim)

        # 1D Local Features (ResNet)
        self.resnet1d_layers = nn.ModuleList([
            ResNetBlock1D(self.hidden_dim, dilation=2 ** min(i, 4))
            for i in range(self.num_res1d)
        ])

        # --- Stage 2: Sequence Context (BiLSTM) ---
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # --- Stage 3: 1D to 2D Projection ---
        dim_1d = self.hidden_dim + self.lstm_hidden * 2
        dim_2d_input = dim_1d * 2

        # 降维层：把拼接后巨大的维度降下来，加入 3 个物理先验通道
        self.proj_2d = nn.Sequential(
            nn.Conv2d(dim_2d_input + 3, self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ELU()
        )

        # --- Stage 4: 2D Mamba Refinement (极速全局修补) ---
        # 抛弃 5 层臃肿的 2D CNN，使用 2 层 2D-Mamba 即可打通全局视野
        self.mamba2d_layers = nn.Sequential(
            MambaBlock2D(self.hidden_dim),
            MambaBlock2D(self.hidden_dim)
        )

        # Output
        self.final_conv = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)

    def forward(self, x, mask=None):
        # x: (B, L, 4)  [One-hot 编码的序列，4个维度代表 A, C, G, U]
        B, L, _ = x.shape

        # ====== 物理先验: 构建基于配对规则的多通道 2D 先验 ======
        A = x[:, :, 0:1]
        C = x[:, :, 1:2]
        G = x[:, :, 2:3]
        U = x[:, :, 3:4]

        AU_pair = torch.bmm(A, U.transpose(1, 2)) + torch.bmm(U, A.transpose(1, 2))
        CG_pair = torch.bmm(C, G.transpose(1, 2)) + torch.bmm(G, C.transpose(1, 2))
        GU_pair = torch.bmm(G, U.transpose(1, 2)) + torch.bmm(U, G.transpose(1, 2))
        prior_2d = torch.stack([AU_pair, CG_pair, GU_pair], dim=1)  # (B, 3, L, L)

        # 1. Embed & 1D ResNet (Local)
        x_emb = self.embedding(x)
        x_local = x_emb.permute(0, 2, 1)
        for layer in self.resnet1d_layers:
            x_local = layer(x_local)
        x_local = x_local.permute(0, 2, 1)  # (B, L, hidden)

        # 2. BiLSTM (Global Context)
        x_lstm, _ = self.lstm(x_local)      # (B, L, 2*lstm_hidden)

        # 3. Combine Local + Global -> (B, L, hidden + 2*lstm_hidden)
        x_1d = torch.cat([x_local, x_lstm], dim=-1)

        # 4. 1D to 2D (Outer Concatenation)
        x_row = x_1d.unsqueeze(2).expand(-1, -1, L, -1)
        x_col = x_1d.unsqueeze(1).expand(-1, L, -1, -1)
        x_2d = torch.cat([x_row, x_col], dim=-1)  # (B, L, L, dim_2d_input)
        x_2d = x_2d.permute(0, 3, 1, 2)           # (B, C, L, L)

        # 把多通道 2D 物理先验拼接到网络特征中
        x_2d = torch.cat([x_2d, prior_2d], dim=1) # 维度变为 dim_2d_input + 3

        # 5. Project & Refine (降维 + 2D Mamba 修图)
        x_2d = self.proj_2d(x_2d)
        x_2d = self.mamba2d_layers(x_2d)

        # 6. Output
        logits = self.final_conv(x_2d).squeeze(1)  # (B, L, L)

        # Symmetrize (保证输出概率矩阵对称，这在接触图预测中非常重要)
        logits = (logits + logits.transpose(1, 2)) / 2
        
        return logits


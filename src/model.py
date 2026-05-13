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
    """2D ResNet Block for Structure Refinement"""
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


class SpotRNA_LSTM_Refined(nn.Module):
    def __init__(self, config):
        super(SpotRNA_LSTM_Refined, self).__init__()
        self.hidden_dim = config.HIDDEN_DIM  
        self.num_res1d = config.RESNET_LAYERS  
        self.lstm_hidden = getattr(config, 'LSTM_HIDDEN', self.hidden_dim)

        self.embedding = nn.Linear(4, self.hidden_dim)

        self.resnet1d_layers = nn.ModuleList([
            ResNetBlock1D(self.hidden_dim, dilation=2 ** min(i, 4))
            for i in range(self.num_res1d)
        ])

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.lstm_hidden,
            num_layers=1,  
            batch_first=True,
            bidirectional=True
        )
            # 【核心新增】MultiheadAttention 用于捕捉长距离(跨界)依赖
        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm_hidden * 2, num_heads=4, batch_first=True
        )

        dim_1d = self.hidden_dim + (self.lstm_hidden * 2) + (self.lstm_hidden * 2)
        dim_2d_input = dim_1d * 2

        self.proj_2d = nn.Sequential(
            nn.Conv2d(dim_2d_input + 3, self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ELU()
        )

        self.resnet2d_layers = nn.Sequential(
            *[ResNetBlock2D(self.hidden_dim) for _ in range(5)]
        )
        self.final_conv = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        
        A = x[:, :, 0:1]
        C = x[:, :, 1:2]
        G = x[:, :, 2:3]
        U = x[:, :, 3:4]
        AU_pair = torch.bmm(A, U.transpose(1, 2)) + torch.bmm(U, A.transpose(1, 2))
        CG_pair = torch.bmm(C, G.transpose(1, 2)) + torch.bmm(G, C.transpose(1, 2))
        GU_pair = torch.bmm(G, U.transpose(1, 2)) + torch.bmm(U, G.transpose(1, 2))
        prior_2d = torch.stack([AU_pair, CG_pair, GU_pair], dim=1)

        x_emb = self.embedding(x)
        x_local = x_emb.permute(0, 2, 1)
        for layer in self.resnet1d_layers:
            x_local = layer(x_local)
        x_local = x_local.permute(0, 2, 1)

        x_lstm, _ = self.lstm(x_local)
        key_padding_mask = (mask.squeeze() == 0) if mask is not None else None
        attn_out, _ = self.attention(x_lstm, x_lstm, x_lstm, key_padding_mask=key_padding_mask)
        
        x_1d = torch.cat([x_local, x_lstm, attn_out], dim=-1)
        x_row = x_1d.unsqueeze(2).expand(-1, -1, L, -1)
        x_col = x_1d.unsqueeze(1).expand(-1, L, -1, -1)
        x_2d = torch.cat([x_row, x_col], dim=-1)
        x_2d = x_2d.permute(0, 3, 1, 2)

        x_2d = torch.cat([x_2d, prior_2d], dim=1)
        x_2d = self.proj_2d(x_2d)
        x_2d = self.resnet2d_layers(x_2d)

        logits = self.final_conv(x_2d).squeeze(1)
        logits = (logits + logits.transpose(1, 2)) / 2

        if mask is not None:
            if mask.dim() == 4:
                mask_1d = mask[:, 0, 0, :]
            elif mask.dim() == 2:
                mask_1d = mask
            else:
                mask_1d = mask.squeeze()
            mask_2d = mask_1d.unsqueeze(2) * mask_1d.unsqueeze(1)
            logits = logits.masked_fill(mask_2d == 0, -1e9)
            
        return logits
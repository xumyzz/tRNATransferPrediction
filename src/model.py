import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResNetBlock1D_GN(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super(ResNetBlock1D_GN, self).__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        num_groups = 8 if channels % 8 == 0 else 4
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        out = F.elu(self.gn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.elu(self.gn2(self.conv2(out)))
        out += residual
        return out


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

class SegmentRelativePosEmb(nn.Module):
    def __init__(self, dim, max_len=1000):
        super().__init__()
        self.emb_tRNA = nn.Embedding(max_len, dim)
        self.emb_siRNA = nn.Embedding(max_len, dim)

    def forward(self, seg_ids):
        B, L = seg_ids.shape
        device = seg_ids.device
        pos = torch.zeros_like(seg_ids, dtype=torch.long)

        for seg_type in [0, 2]:
            type_mask = (seg_ids == seg_type).long()
            cumsum = torch.cumsum(type_mask, dim=1)
            domain_pos = (cumsum - 1) * type_mask
            pos = pos + domain_pos

        emb = torch.zeros(B, L, self.emb_tRNA.embedding_dim, device=device)
        emb[seg_ids == 0] = self.emb_tRNA(pos[seg_ids == 0])
        emb[seg_ids == 2] = self.emb_siRNA(pos[seg_ids == 2])
        return emb

class SpotRNA_LSTM_Refined(nn.Module):
    """
    老模型架构（纯 LSTM，无 Attention 机制）
    用于加载最早训练好的权重 (tRNA_Finetune/model_best.pth)
    """
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

        # 这里匹配的是老权重的维度: 387 = (64 + 128) * 2 + 3
        dim_1d = self.hidden_dim + self.lstm_hidden * 2
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
        
        # 提取基于互补规则的物理先验知识
        A = x[:, :, 0:1]
        C = x[:, :, 1:2]
        G = x[:, :, 2:3]
        U = x[:, :, 3:4]
        AU_pair = torch.bmm(A, U.transpose(1, 2)) + torch.bmm(U, A.transpose(1, 2))
        CG_pair = torch.bmm(C, G.transpose(1, 2)) + torch.bmm(G, C.transpose(1, 2))
        GU_pair = torch.bmm(G, U.transpose(1, 2)) + torch.bmm(U, G.transpose(1, 2))
        prior_2d = torch.stack([AU_pair, CG_pair, GU_pair], dim=1)

        # 1D 特征提取
        x_emb = self.embedding(x)
        x_local = x_emb.permute(0, 2, 1)
        for layer in self.resnet1d_layers:
            x_local = layer(x_local)
        x_local = x_local.permute(0, 2, 1)

        # 序列上下文提取
        x_lstm, _ = self.lstm(x_local)

        # 特征融合（只有 Local ResNet + Context LSTM，没有 Attention）
        x_1d = torch.cat([x_local, x_lstm], dim=-1)

        # 转换为 2D Map
        # 学两个投影矩阵，而不是暴力展开
        self.proj_row = nn.Linear(dim_1d, dim_1d)
        self.proj_col = nn.Linear(dim_1d, dim_1d)
        x_2d = self.proj_row(x_1d).unsqueeze(2) + self.proj_col(x_1d).unsqueeze(1)
        x_2d = x_2d.permute(0, 3, 1, 2)

        # 加入先验特征并通过 2D ResNet
        x_2d = torch.cat([x_2d, prior_2d], dim=1)
        x_2d = self.proj_2d(x_2d)
        x_2d = self.resnet2d_layers(x_2d)

        logits = self.final_conv(x_2d).squeeze(1)
        logits = (logits + logits.transpose(1, 2)) / 2

        # 屏蔽无效区域（针对 batch padding）
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
        
class SpotRNA_LSTM_Refined_Attention(nn.Module):
    def __init__(self, config):
        super(SpotRNA_LSTM_Refined_Attention, self).__init__()
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

class SpotRNA_LSTM_Refined_BPPM(nn.Module):
    """
    升级版模型架构（引入 BPPM 热力学通道）
    在老模型基础上，增加一个额外的 2D 输入通道 (bppm)
    """
    def __init__(self, config):
        super(SpotRNA_LSTM_Refined_BPPM, self).__init__()
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

        dim_1d = self.hidden_dim + self.lstm_hidden * 2
        dim_2d_input = dim_1d * 2

        # 【核心修改点】
        # 以前是 dim_2d_input + 3 (AU/CG/GU先验)
        # 现在多了一个 bppm 通道，所以输入维度变成了 dim_2d_input + 4
        self.proj_2d = nn.Sequential(
            nn.Conv2d(dim_2d_input + 4, self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ELU()
        )

        self.resnet2d_layers = nn.Sequential(
            *[ResNetBlock2D(self.hidden_dim) for _ in range(5)]
        )

        self.final_conv = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)

    def forward(self, x, bppm=None, mask=None):
        """
        新增参数 bppm: shape (Batch, 1, L, L), 也就是计算出的配分函数矩阵
        """
        B, L, _ = x.shape
        
        # 1. 提取基于互补规则的物理先验知识
        A = x[:, :, 0:1]
        C = x[:, :, 1:2]
        G = x[:, :, 2:3]
        U = x[:, :, 3:4]
        AU_pair = torch.bmm(A, U.transpose(1, 2)) + torch.bmm(U, A.transpose(1, 2))
        CG_pair = torch.bmm(C, G.transpose(1, 2)) + torch.bmm(G, C.transpose(1, 2))
        GU_pair = torch.bmm(G, U.transpose(1, 2)) + torch.bmm(U, G.transpose(1, 2))
        
        # (Batch, 3, L, L)
        prior_2d = torch.stack([AU_pair, CG_pair, GU_pair], dim=1) 
        
        # 【拼接 BPPM 通道】
        if bppm is not None:
            # bppm 通常是 (Batch, L, L)，我们需要它变成 (Batch, 1, L, L)
            if len(bppm.shape) == 3:
                bppm = bppm.unsqueeze(1)
            # prior_2d 变成 (Batch, 4, L, L)
            prior_2d = torch.cat([prior_2d, bppm], dim=1)
        else:
            # 为了防止你忘了传，如果没传，就默认全零（退化为原模型，但通道数要对上）
            dummy_bppm = torch.zeros((B, 1, L, L), device=x.device)
            prior_2d = torch.cat([prior_2d, dummy_bppm], dim=1)

        # 2. 1D 特征提取
        x_emb = self.embedding(x)
        x_local = x_emb.permute(0, 2, 1)
        for layer in self.resnet1d_layers:
            x_local = layer(x_local)
        x_local = x_local.permute(0, 2, 1)

        # 3. 序列上下文提取
        x_lstm, _ = self.lstm(x_local)

        # 4. 特征融合
        x_1d = torch.cat([x_local, x_lstm], dim=-1)

        # 5. 转换为 2D Map
        x_row = x_1d.unsqueeze(2).expand(-1, -1, L, -1)
        x_col = x_1d.unsqueeze(1).expand(-1, L, -1, -1)
        x_2d = torch.cat([x_row, x_col], dim=-1)
        x_2d = x_2d.permute(0, 3, 1, 2)

        # 6. 加入先验特征(含 BPPM)并通过 2D ResNet
        x_2d = torch.cat([x_2d, prior_2d], dim=1)
        x_2d = self.proj_2d(x_2d)
        x_2d = self.resnet2d_layers(x_2d)

        logits = self.final_conv(x_2d).squeeze(1)
        logits = (logits + logits.transpose(1, 2)) / 2

        # 7. 屏蔽无效区域
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

class SpotRNA_Chimeric(nn.Module):
    """
    针对 tRNA-siRNA/miRNA 嵌合RNA的特化模型
    基于你原来的 SpotRNA_LSTM_Refined_Attention 修改

    使用方式：
        model = SpotRNA_Chimeric(config)
        # domain_split_idx: tRNA部分的长度（siRNA从该索引开始）
        # 例如标准tRNA长度为76，则 siRNA 从第76个碱基开始
        logits = model(x, domain_split_idx=76, mask=mask)
    """
    def __init__(self, config):
        super(SpotRNA_Chimeric, self).__init__()
        self.hidden_dim = config.HIDDEN_DIM
        self.num_res1d = config.RESNET_LAYERS
        self.lstm_hidden = getattr(config, 'LSTM_HIDDEN', self.hidden_dim)

        # 默认tRNA长度（当用户不传入domain_split_idx时使用，建议你在config里设成你的实际长度）
        self.default_trna_len = getattr(config, 'DEFAULT_TRNA_LEN', 76)

        self.embedding = nn.Linear(4, self.hidden_dim)

        # 1D特征提取：使用GroupNorm版ResNet
        self.resnet1d_layers = nn.ModuleList([
            ResNetBlock1D_GN(self.hidden_dim, dilation=2 ** min(i, 4))
            for i in range(self.num_res1d)
        ])

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Attention保留，但会在forward中被域掩码限制，阻止跨域注意力泄漏
        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm_hidden * 2, num_heads=4, batch_first=True
        )

        # 分段相对位置编码
        self.seg_pos_emb = SegmentRelativePosEmb(dim=self.lstm_hidden * 2, max_len=1000)

        # 特征维度计算（与原版一致：local + lstm + attn）
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

    # -------------------- 域掩码生成工具函数 --------------------
    def _generate_seg_ids(self, L, domain_split_idx, device):
        """
        根据 domain_split_idx 自动生成域标签
        domain_split_idx: (B,) LongTensor, 表示 siRNA/miRNA 起始索引（0-based）
        返回: (B, L) LongTensor, 0=tRNA, 2=siRNA
        """
        B = domain_split_idx.size(0)
        seg_ids = torch.zeros(B, L, dtype=torch.long, device=device)
        for b in range(B):
            split = domain_split_idx[b].item()
            split = max(1, min(split, L - 1))  # 安全检查
            seg_ids[b, split:] = 2               # 后半部分标记为siRNA域
        return seg_ids

    def _create_domain_attention_mask(self, seg_ids, pad_mask=None):
        """
        生成Attention掩码：严格禁止 tRNA(0) 与 siRNA(2) 之间的注意力
        返回: (B, L, L) BoolTensor, True = 禁止（mask掉）
        """
        seg_i = seg_ids.unsqueeze(2)  # (B, L, 1)
        seg_j = seg_ids.unsqueeze(1)  # (B, 1, L)
        cross_domain = (seg_i != seg_j)  # True=跨域

        if pad_mask is not None:
            # pad_mask: (B, L) bool, True=padding位置
            pad_mask_2d = pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2)
            cross_domain = cross_domain | pad_mask_2d
        return cross_domain  # True=禁止

    def _create_domain_2d_mask(self, seg_ids):
        """
        生成2D接触图硬掩码：只允许同一域内配对
        返回: (B, L, L) BoolTensor, True = 允许配对
        """
        seg_i = seg_ids.unsqueeze(2)
        seg_j = seg_ids.unsqueeze(1)
        same_domain = (seg_i == seg_j)
        return same_domain

    # -------------------- 前向传播 --------------------
    def forward(self, x, domain_split_idx=None, mask=None, return_seg_ids=False):
        """
        参数:
            x: (B, L, 4) one-hot序列
            domain_split_idx: int 或 (B,) LongTensor 或 None
                - 必须传入！表示每条序列中 tRNA 部分的长度（siRNA从该索引开始）
                - 例如 torch.tensor([76, 76, 76]) 或简单整数 76
                - 如果不传，默认使用 config.DEFAULT_TRNA_LEN（建议你在config里设置好）
            mask: 你的原始mask（支持2D/4D，与原版兼容）
            return_seg_ids: bool，调试时设为True可返回自动生成的域标签
        """
        B, L, _ = x.shape
        device = x.device

        # ---------- 0. 确定域分割点（没有预标注数据，由传入的整数自动推导）----------
        if domain_split_idx is None:
            # 【重要】fallback到默认长度。强烈建议你在config里设对！
            domain_split_idx = torch.full((B,), self.default_trna_len, dtype=torch.long, device=device)
        elif isinstance(domain_split_idx, int):
            domain_split_idx = torch.full((B,), domain_split_idx, dtype=torch.long, device=device)
        elif domain_split_idx.dim() == 0:
            domain_split_idx = domain_split_idx.unsqueeze(0).expand(B)

        seg_ids = self._generate_seg_ids(L, domain_split_idx, device)

        # ---------- 1. 物理先验（AU/CG/GU互补规则）----------
        A = x[:, :, 0:1]
        C = x[:, :, 1:2]
        G = x[:, :, 2:3]
        U = x[:, :, 3:4]
        AU_pair = torch.bmm(A, U.transpose(1, 2)) + torch.bmm(U, A.transpose(1, 2))
        CG_pair = torch.bmm(C, G.transpose(1, 2)) + torch.bmm(G, C.transpose(1, 2))
        GU_pair = torch.bmm(G, U.transpose(1, 2)) + torch.bmm(U, G.transpose(1, 2))
        prior_2d = torch.stack([AU_pair, CG_pair, GU_pair], dim=1)

        # ---------- 2. 1D 局部特征 ----------
        x_emb = self.embedding(x)
        x_local = x_emb.permute(0, 2, 1)
        for layer in self.resnet1d_layers:
            x_local = layer(x_local)
        x_local = x_local.permute(0, 2, 1)

        # ---------- 3. LSTM 序列上下文 ----------
        x_lstm, _ = self.lstm(x_local)

        # ---------- 4. 分段位置编码注入（关键！）----------
        seg_pos = self.seg_pos_emb(seg_ids)
        x_lstm_pos = x_lstm + seg_pos  # 广播加法，让模型感知"我在哪个域的哪个相对位置"

        # ---------- 5. 【核心修改】域受限 Attention ----------
        # 处理原始padding mask（兼容你原来的多维度mask逻辑）
        if mask is not None:
            if mask.dim() == 4:
                mask_1d = mask[:, 0, 0, :]
            elif mask.dim() == 2:
                mask_1d = mask
            else:
                mask_1d = mask.squeeze()
            key_padding_mask = (mask_1d == 0)
        else:
            key_padding_mask = None

        # 生成跨域注意力掩码：True=禁止
        domain_attn_mask = self._create_domain_attention_mask(seg_ids, pad_mask=key_padding_mask)

        # Attention前向：硬阻断跨域注意力，tRNA内部和siRNA内部的长程依赖完全保留
        attn_out, _ = self.attention(
            x_lstm_pos, x_lstm_pos, x_lstm_pos,
            key_padding_mask=key_padding_mask,
            attn_mask=domain_attn_mask
        )

        # ---------- 6. 特征融合（与原版维度一致）----------
        x_1d = torch.cat([x_local, x_lstm_pos, attn_out], dim=-1)

        # ---------- 7. 构建 2D Map ----------
        x_row = x_1d.unsqueeze(2).expand(-1, -1, L, -1)
        x_col = x_1d.unsqueeze(1).expand(-1, L, -1, -1)
        x_2d = torch.cat([x_row, x_col], dim=-1)
        x_2d = x_2d.permute(0, 3, 1, 2)

        # ---------- 8. 2D ResNet ----------
        x_2d = torch.cat([x_2d, prior_2d], dim=1)
        x_2d = self.proj_2d(x_2d)
        x_2d = self.resnet2d_layers(x_2d)

        logits = self.final_conv(x_2d).squeeze(1)  # (B, L, L)

        # ---------- 9. 对称化 ----------
        logits = (logits + logits.transpose(1, 2)) / 2

        # ---------- 10. 【核心修改】对角线硬约束（禁止自身配对）----------
        diag_mask = torch.eye(L, device=device).unsqueeze(0).bool()
        logits = logits.masked_fill(diag_mask, -1e9)

        # ---------- 11. 【核心修改】跨域2D硬掩码（禁止tRNA与siRNA配对）----------
        domain_2d_mask = self._create_domain_2d_mask(seg_ids)  # True=允许
        logits = logits.masked_fill(~domain_2d_mask, -1e9)

        # ---------- 12. 原始padding mask处理（兼容原版）----------
        if mask is not None:
            if mask.dim() == 4:
                mask_1d = mask[:, 0, 0, :]
            elif mask.dim() == 2:
                mask_1d = mask
            else:
                mask_1d = mask.squeeze()
            mask_2d = mask_1d.unsqueeze(2) * mask_1d.unsqueeze(1)
            logits = logits.masked_fill(mask_2d == 0, -1e9)

        if return_seg_ids:
            return logits, seg_ids
        return logits

class SpotRNA_LSTM_Refined_BPPM_Chimeric(nn.Module):
    def __init__(self, config):
        super(SpotRNA_LSTM_Refined_BPPM_Chimeric, self).__init__()
        self.hidden_dim = config.HIDDEN_DIM
        self.num_res1d = config.RESNET_LAYERS
        self.lstm_hidden = getattr(config, 'LSTM_HIDDEN', self.hidden_dim)
        

        self.embedding = nn.Linear(4, self.hidden_dim)

        self.resnet1d_layers = nn.ModuleList([
            ResNetBlock1D_GN(self.hidden_dim, dilation=2 ** min(i, 4))
            for i in range(self.num_res1d)
        ])

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm_hidden * 2, num_heads=4, batch_first=True
        )

        self.seg_pos_emb = SegmentRelativePosEmb(dim=self.lstm_hidden * 2, max_len=1000)

        dim_1d = self.hidden_dim + (self.lstm_hidden * 2) + (self.lstm_hidden * 2)
        dim_2d_input = dim_1d * 2

        self.proj_2d = nn.Sequential(
            nn.Conv2d(dim_2d_input + 4, self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ELU()
        )

        self.resnet2d_layers = nn.Sequential(
            *[ResNetBlock2D(self.hidden_dim) for _ in range(5)]
        )
        self.final_conv = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)

    def _generate_seg_ids_triple(self, B, L, trna_5end_len, trna_3end_len, device):
        """
        三段式域标签：5'tRNA(域0) - miRNA前体(域2) - 3'tRNA(域0)
        """
        seg_ids = torch.zeros(B, L, dtype=torch.long, device=device)
        
        if trna_5end_len is None or trna_3end_len is None:
            return seg_ids  # 全0，普通RNA模式
        
        for b in range(B):
            five_len = trna_5end_len[b].item()
            three_len = trna_3end_len[b].item()
            
            if five_len + three_len >= L:
                continue
            
            mirna_start = five_len
            mirna_end = L - three_len
            
            if mirna_start < mirna_end:
                seg_ids[b, mirna_start:mirna_end] = 2
        
        return seg_ids

    def _create_domain_attention_mask(self, seg_ids):
        seg_i = seg_ids.unsqueeze(2)
        seg_j = seg_ids.unsqueeze(1)
        cross_domain = (seg_i != seg_j)
        return cross_domain

    def _create_domain_2d_mask(self, seg_ids):
        seg_i = seg_ids.unsqueeze(2)
        seg_j = seg_ids.unsqueeze(1)
        same_domain = (seg_i == seg_j)
        return same_domain

    def forward(self, x, bppm=None, mask=None, 
                trna_5end_len=None, trna_3end_len=None, 
                return_seg_ids=False):
        B, L, _ = x.shape
        device = x.device

               # 0. 三段式域标签生成
        if trna_5end_len is None or trna_3end_len is None:
            # 普通RNA预训练模式：全序列同域
            seg_ids = torch.zeros(B, L, dtype=torch.long, device=device)
        else:
            if isinstance(trna_5end_len, int):
                trna_5end_len = torch.full((B,), trna_5end_len, dtype=torch.long, device=device)
            if isinstance(trna_3end_len, int):
                trna_3end_len = torch.full((B,), trna_3end_len, dtype=torch.long, device=device)
            seg_ids = self._generate_seg_ids_triple(B, L, trna_5end_len, trna_3end_len, device)

        # 1. 物理先验（AU/CG/GU互补规则）
        A = x[:, :, 0:1]
        C = x[:, :, 1:2]
        G = x[:, :, 2:3]
        U = x[:, :, 3:4]
        AU_pair = torch.bmm(A, U.transpose(1, 2)) + torch.bmm(U, A.transpose(1, 2))
        CG_pair = torch.bmm(C, G.transpose(1, 2)) + torch.bmm(G, C.transpose(1, 2))
        GU_pair = torch.bmm(G, U.transpose(1, 2)) + torch.bmm(U, G.transpose(1, 2))
        prior_2d = torch.stack([AU_pair, CG_pair, GU_pair], dim=1)

        # 2. BPPM 通道（与你原版完全一致）
        if bppm is not None:
            if len(bppm.shape) == 3:
                bppm = bppm.unsqueeze(1)
            prior_2d = torch.cat([prior_2d, bppm], dim=1)
        else:
            dummy_bppm = torch.zeros((B, 1, L, L), device=device)
            prior_2d = torch.cat([prior_2d, dummy_bppm], dim=1)

        # 3. 1D 局部特征
        x_emb = self.embedding(x)
        x_local = x_emb.permute(0, 2, 1)
        for layer in self.resnet1d_layers:
            x_local = layer(x_local)
        x_local = x_local.permute(0, 2, 1)

        # 4. LSTM
        x_lstm, _ = self.lstm(x_local)

        # 5. 分段位置编码注入
        seg_pos = self.seg_pos_emb(seg_ids)
        x_lstm_pos = x_lstm + seg_pos

        # ==================== 【终极修复】MHA 输入预处理 ====================
        # 提取 1D padding mask
        if mask is not None:
            if mask.dim() == 4:
                mask_1d = mask[:, 0, 0, :]
            elif mask.dim() == 2:
                mask_1d = mask
            else:
                mask_1d = mask.squeeze()
            if mask_1d.dim() == 1:
                mask_1d = mask_1d.unsqueeze(0).expand(B, -1)
        else:
            mask_1d = torch.ones(B, L, device=device)

        # MHA 前把 padding 位置特征清零（避免 padding 参与 attention）
        x_lstm_pos = x_lstm_pos * mask_1d.unsqueeze(-1).float()

        # ==================== 【终极修复】只用 Float Additive Mask ====================
        # 构造 float bias mask: 允许=0, 禁止=-1e9
        # PyTorch MHA 对 float mask 的 backward 远稳定于 bool mask
        attn_bias = torch.zeros(B, L, L, device=device)

        # 跨域禁止
        domain_mask = self._create_domain_attention_mask(seg_ids)
        attn_bias[domain_mask] = -1e9

        # padding 禁止（query 或 key 任一 padding 就禁止）
        pad_mask_1d = (mask_1d == 0)
        pad_mask_2d = pad_mask_1d.unsqueeze(1) | pad_mask_1d.unsqueeze(2)
        attn_bias[pad_mask_2d] = -1e9

        # diagonal 强制允许（自己看到自己，防止 softmax NaN）
        eye = torch.eye(L, device=device).unsqueeze(0).expand(B, -1, -1).bool()  # (B, L, L)
        attn_bias = attn_bias.masked_fill(eye, 0.0)
        row_all_forbidden = (attn_bias <= -1e8).all(dim=-1, keepdim=True)
        attn_bias = torch.where(row_all_forbidden & ~eye, torch.zeros_like(attn_bias), attn_bias)
        # 扩展为 (B*num_heads, L, L)
        num_heads = self.attention.num_heads
        attn_bias = attn_bias.repeat_interleave(num_heads, dim=0)

        # 【关键】只用 attn_mask，不用 key_padding_mask，避免两者混用导致 NaN
        attn_out, _ = self.attention(
            x_lstm_pos, x_lstm_pos, x_lstm_pos,
            attn_mask=attn_bias
        )

        # 7. 特征融合
        x_1d = torch.cat([x_local, x_lstm_pos, attn_out], dim=-1)

        # 8. 2D Map
        x_row = x_1d.unsqueeze(2).expand(-1, -1, L, -1)
        x_col = x_1d.unsqueeze(1).expand(-1, L, -1, -1)
        x_2d = torch.cat([x_row, x_col], dim=-1)
        x_2d = x_2d.permute(0, 3, 1, 2)

        # 9. 2D ResNet
        x_2d = torch.cat([x_2d, prior_2d], dim=1)
        x_2d = self.proj_2d(x_2d)
        x_2d = self.resnet2d_layers(x_2d)

        logits = self.final_conv(x_2d).squeeze(1)

        # 10. 对称化
        logits = (logits + logits.transpose(1, 2)) / 2

        # 11. 对角线硬约束（禁止自身配对）
        diag_mask = torch.eye(L, device=device).unsqueeze(0).bool()
        logits = logits.masked_fill(diag_mask, -1e9)

        # 12. 跨域 2D 硬掩码（禁止 tRNA↔siRNA 配对）
        domain_2d_mask = self._create_domain_2d_mask(seg_ids)
        logits = logits.masked_fill(~domain_2d_mask, -1e9)

        # 13. 原始 padding mask
        if mask is not None:
            if mask.dim() == 4:
                mask_1d_2d = mask[:, 0, 0, :]
            elif mask.dim() == 2:
                mask_1d_2d = mask
            else:
                mask_1d_2d = mask.squeeze()
            mask_2d = mask_1d_2d.unsqueeze(2) * mask_1d_2d.unsqueeze(1)
            logits = logits.masked_fill(mask_2d == 0, -1e9)

        if return_seg_ids:
            return logits, seg_ids
        return logits
# ==================== 【修改点4】配套损失函数 ====================
class ChimericRNALoss(nn.Module):
    """
    嵌合RNA专用损失：
    1. Focal Loss（解决配对极度稀疏问题）
    2. Dice Loss（促进茎区连续性）
    3. 对角线惩罚（自身不配對）
    4. 跨域惩罚（tRNA与siRNA之间预测配对的额外重罚）
    """
    def __init__(self, alpha=0.25, gamma=2.0, w_dice=0.3, lambda_cross=5.0, lambda_diag=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.w_dice = w_dice
        self.lambda_cross = lambda_cross   # 跨域惩罚权重，建议5~20
        self.lambda_diag = lambda_diag

    def forward(self, logits, targets, seg_ids=None, mask_2d=None):
        B, L, _ = logits.shape
        device = logits.device

        # ---- Focal Loss ----
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce

        # ---- Dice Loss ----
        pred_prob = torch.sigmoid(logits)
        intersection = (pred_prob * targets).sum(dim=(-2, -1))
        union = (pred_prob ** 2).sum(dim=(-2, -1)) + (targets ** 2).sum(dim=(-2, -1))
        dice = 1 - (2 * intersection + 1e-6) / (union + 1e-6)

        loss = focal.mean() + self.w_dice * dice.mean()

        # ---- 对角线惩罚 ----
        diag_mask = torch.eye(L, device=device).unsqueeze(0)
        diag_penalty = (pred_prob * diag_mask).sum(dim=(-2, -1))
        loss = loss + self.lambda_diag * diag_penalty.mean()

        # ---- 跨域惩罚（关键！）----
        if seg_ids is not None:
            seg_i = seg_ids.unsqueeze(2)
            seg_j = seg_ids.unsqueeze(1)
            cross_domain = (seg_i != seg_j).float()          # 1=跨域，0=域内
            cross_penalty = (pred_prob * cross_domain).sum(dim=(-2, -1))
            loss = loss + self.lambda_cross * cross_penalty.mean()

        # ---- mask过滤 ----
        if mask_2d is not None:
            loss = loss * mask_2d.float()

        return loss.mean()

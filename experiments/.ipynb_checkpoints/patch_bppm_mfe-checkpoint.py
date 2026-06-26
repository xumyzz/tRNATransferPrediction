#!/usr/bin/env python3
"""
一键 patch：BPPM + MFE 通道启用
=================================
在服务器上运行一次，自动修改 src/dataset.py + src/model.py + 训练/评测脚本。

运行前确认 ViennaRNA 可用：
    python -c "import RNA; f=RNA.fold_compound('GGGCCC'); f.pf(); print('OK')"

然后：
    python experiments/patch_bppm_mfe.py
"""

import os
import sys
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

def backup(path):
    """备份文件"""
    bak = Path(str(path) + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  backup: {bak}")

def patch_file(path, old, new, count=1):
    """在文件中替换文本"""
    content = Path(path).read_text()
    if old not in content:
        print(f"  [skip] pattern not found in {path.name}")
        return False
    content = content.replace(old, new, count)
    Path(path).write_text(content)
    print(f"  patched: {path.name}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Patch 1: src/dataset.py — 加 get_mfe_contact + collate_pad_v2
# ═══════════════════════════════════════════════════════════════════════════

def patch_dataset():
    path = PROJECT / "src" / "dataset.py"
    backup(path)

    # 1a. 在 get_bppm_matrix 后面加 get_mfe_contact
    old = '''    return matrix


# --- 3. 更新后的 collate_pad ---'''
    new = '''    return matrix


def get_mfe_contact(seq):
    """
    用 ViennaRNA RNAfold 计算 MFE 结构的二值接触图 (L, L) int8
    ViennaRNA 不可用时返回全零矩阵。
    """
    L = len(seq)
    safe_seq = seq.replace('N', 'A')
    try:
        fc = RNA.fold_compound(safe_seq)
        _, mfe = fc.mfe()
    except Exception:
        return np.zeros((L, L), dtype=np.float32)

    contact = np.zeros((L, L), dtype=np.float32)
    stacks = {'(': [], '[': [], '{': [], '<': []}
    pairs_map = {')': '(', ']': '[', '}': '{', '>': '<'}
    for i, c in enumerate(mfe[:L]):
        if c in stacks:
            stacks[c].append(i)
        elif c in pairs_map and stacks[pairs_map[c]]:
            j = stacks[pairs_map[c]].pop()
            contact[i, j] = contact[j, i] = 1.0
    return contact


# --- 3. 更新后的 collate_pad ---'''
    patch_file(path, old, new)

    # 1b. 在文件末尾追加 collate_pad_v2
    end_marker = "def get_name(self, idx):"
    if end_marker not in Path(path).read_text():
        print("  [warn] 未找到 get_name, 尝试查找文件末尾")
        end_marker = "return self.names[idx]"

    old = end_marker
    new = end_marker + '''
    # 返回名称（用于查预计算特征字典）


def collate_pad_v2(batch):
    """
    collate_pad 的 v2 版本：多返回一个 mfe_contact 通道。
    返回 5 元组: (seqs, bppms, mfe_contacts, labels, masks)
    """
    max_len = max([x[0].shape[0] for x in batch])
    b_size = len(batch)

    seqs = torch.zeros(b_size, max_len, 4)
    bppms = torch.zeros(b_size, max_len, max_len)
    mfe_contacts = torch.zeros(b_size, max_len, max_len)
    labels = torch.zeros(b_size, max_len, max_len)
    masks = torch.zeros(b_size, max_len)

    for i, (s, b, l) in enumerate(batch):
        n = s.shape[0]
        seqs[i, :n] = s
        bppms[i, :n, :n] = b
        labels[i, :n, :n] = l
        masks[i, :n] = 1.0

        # 从 one-hot 重建序列字符串，计算 MFE
        idx_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
        seq_str = ''.join(idx_to_base.get(int(s[j].argmax().item()), 'A') for j in range(n))
        mfe_contacts[i, :n, :n] = torch.from_numpy(get_mfe_contact(seq_str))

    return seqs, bppms, mfe_contacts, labels, masks
'''
    patch_file(path, old, new)
    print("  [done] dataset.py\n")


# ═══════════════════════════════════════════════════════════════════════════
# Patch 2: src/model.py — SpotRNA_LSTM_Refined_BPPM_Chimeric 加 MFE 通道
# ═══════════════════════════════════════════════════════════════════════════

def patch_model():
    path = PROJECT / "src" / "model.py"
    backup(path)

    # 2a. proj_2d 输入维度: +4 → +5
    # 在 Chimeric 类的 __init__ 中，只改 Chimeric 类里的那个 +4
    # 注意：BPPM 老版也有 +4，不要改错
    content = Path(path).read_text()

    # 找到 Chimeric 类的 proj_2d 定义（它在 dim_1d = self.hidden_dim + self.attn_dim + self.attn_dim 之后）
    old = '''        dim_1d = self.hidden_dim + self.attn_dim + self.attn_dim
        dim_2d_input = dim_1d * 2

        self.proj_2d = nn.Sequential(
            nn.Conv2d(dim_2d_input + 4, self.hidden_dim, kernel_size=1),'''

    new = '''        dim_1d = self.hidden_dim + self.attn_dim + self.attn_dim
        dim_2d_input = dim_1d * 2

        self.proj_2d = nn.Sequential(
            nn.Conv2d(dim_2d_input + 5, self.hidden_dim, kernel_size=1),'''
    patch_file(path, old, new)

    # 2b. forward 签名: 加 mfe_contact 参数
    old = '''    def forward(self, x, bppm=None, mask=None,
                trna_5end_len=None, trna_3end_len=None,
                return_seg_ids=False):'''

    new = '''    def forward(self, x, bppm=None, mfe_contact=None, mask=None,
                trna_5end_len=None, trna_3end_len=None,
                return_seg_ids=False):'''
    patch_file(path, old, new)

    # 2c. BPPM 拼接后面加 MFE 拼接
    old = '''        # ── 2. BPPM ──
        if bppm is not None:
            if len(bppm.shape) == 3:
                bppm = bppm.unsqueeze(1)
            prior_2d = torch.cat([prior_2d, bppm], dim=1)
        else:
            dummy_bppm = torch.zeros((B, 1, L, L), device=device)
            prior_2d = torch.cat([prior_2d, dummy_bppm], dim=1)'''

    new = '''        # ── 2. BPPM ──
        if bppm is not None:
            if len(bppm.shape) == 3:
                bppm = bppm.unsqueeze(1)
            prior_2d = torch.cat([prior_2d, bppm], dim=1)
        else:
            dummy_bppm = torch.zeros((B, 1, L, L), device=device)
            prior_2d = torch.cat([prior_2d, dummy_bppm], dim=1)

        # ── 2b. MFE contact (新增通道) ──
        if mfe_contact is not None:
            if len(mfe_contact.shape) == 3:
                mfe_contact = mfe_contact.unsqueeze(1)
            prior_2d = torch.cat([prior_2d, mfe_contact], dim=1)
        else:
            dummy_mfe = torch.zeros((B, 1, L, L), device=device)
            prior_2d = torch.cat([prior_2d, dummy_mfe], dim=1)'''
    patch_file(path, old, new)

    print("  [done] model.py\n")


# ═══════════════════════════════════════════════════════════════════════════
# Patch 3: experiments/train_v2_standard.py — 用 v2 collate + 传 mfe
# ═══════════════════════════════════════════════════════════════════════════

def patch_train_script():
    path = PROJECT / "experiments" / "train_v2_standard.py"
    backup(path)

    # 3a. import
    old = "from src.dataset import MultiFileDatasetUpgrade, collate_pad"
    new = "from src.dataset import MultiFileDatasetUpgrade, collate_pad_v2"
    patch_file(path, old, new)

    # 3b. DataLoader — train
    old = "train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,\n                              collate_fn=collate_pad, num_workers=args.num_workers)"
    new = "train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,\n                              collate_fn=collate_pad_v2, num_workers=args.num_workers)"
    patch_file(path, old, new)

    # 3c. DataLoader — val
    old = "val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,\n                            collate_fn=collate_pad, num_workers=args.num_workers)"
    new = "val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,\n                            collate_fn=collate_pad_v2, num_workers=args.num_workers)"
    patch_file(path, old, new)

    # 3d. 训练循环中解包 seqs, bppms, labels, masks → 5 元组
    old = "for step, (seqs, bppms, labels, masks) in enumerate(train_loader):"
    new = "for step, (seqs, bppms, mfe_contacts, labels, masks) in enumerate(train_loader):"
    patch_file(path, old, new)

    # 3e. 训练中传 mfe_contact 到模型
    old = "            logits = model(seqs, bppm=bppms, mask=masks,\n                           trna_5end_len=None, trna_3end_len=None)"
    new = "            logits = model(seqs, bppm=bppms, mfe_contact=mfe_contacts, mask=masks,\n                           trna_5end_len=None, trna_3end_len=None)"
    patch_file(path, old, new)

    # 3f. 验证循环中解包 + 传 mfe
    old = "for seqs, bppms, labels, masks in val_loader:"
    new = "for seqs, bppms, mfe_contacts, labels, masks in val_loader:"
    patch_file(path, old, new)

    old = "                logits = model(seqs, bppm=bppms, mask=masks,\n                               trna_5end_len=None, trna_3end_len=None)"
    new = "                logits = model(seqs, bppm=bppms, mfe_contact=mfe_contacts, mask=masks,\n                               trna_5end_len=None, trna_3end_len=None)"
    patch_file(path, old, new)

    print("  [done] train_v2_standard.py\n")


# ═══════════════════════════════════════════════════════════════════════════
# Patch 4: experiments/ts0_evaluate.py — 推理时传 MFE
# ═══════════════════════════════════════════════════════════════════════════

def patch_eval_script():
    path = PROJECT / "experiments" / "ts0_evaluate.py"
    backup(path)

    # 4a. ModelRunner.predict — 计算并传入 MFE
    old = '''        mask = torch.ones(1, L, dtype=torch.float32).to(self.device)

        logits = self.model(x, bppm=bppm, mask=mask,
                            trna_5end_len=None, trna_3end_len=None)'''

    new = '''        mask = torch.ones(1, L, dtype=torch.float32).to(self.device)

        # MFE contact
        if self.use_bppm:
            try:
                fc = _RNA.fold_compound(sample.sequence.replace('N', 'A'))
                _, mfe_dbn = fc.mfe()
                mfe_contact = np.zeros((L, L), dtype=np.float32)
                stacks = {'(': [], '[': [], '{': [], '<': []}
                pairs_map = {')': '(', ']': '[', '}': '{', '>': '<'}
                for i, c in enumerate(mfe_dbn[:L]):
                    if c in stacks:
                        stacks[c].append(i)
                    elif c in pairs_map and stacks[pairs_map[c]]:
                        j = stacks[pairs_map[c]].pop()
                        mfe_contact[i, j] = mfe_contact[j, i] = 1.0
                mfe_t = torch.tensor(mfe_contact, dtype=torch.float32).unsqueeze(0).to(self.device)
            except Exception:
                mfe_t = None
        else:
            mfe_t = None

        logits = self.model(x, bppm=bppm, mfe_contact=mfe_t, mask=mask,
                            trna_5end_len=None, trna_3end_len=None)'''

    patch_file(path, old, new)

    print("  [done] ts0_evaluate.py\n")


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 先检查 ViennaRNA
    try:
        import RNA
        fc = RNA.fold_compound("GGGCCC")
        fc.pf()
        print("[check] ViennaRNA OK\n")
    except Exception as e:
        print(f"[ERROR] ViennaRNA 不可用: {e}")
        print("请先: pip install viennarna")
        sys.exit(1)

    print("开始 patch...\n")
    patch_dataset()
    patch_model()
    patch_train_script()
    patch_eval_script()

    print("=" * 50)
    print("  Patch 完成！")
    print()
    print("验证 patch:")
    print("  grep 'mfe_contact' src/model.py | head -3")
    print("  grep 'collate_pad_v2' experiments/train_v2_standard.py | head -3")
    print()
    print("训练:")
    print("  python experiments/train_v2_standard.py \\")
    print("      --train_dir data/TR0 --val_dir data/VL0 \\")
    print("      --save_dir checkpoints_v2_tr0_bppm_mfe \\")
    print("      --epochs 50 --device cuda:0")
    print("=" * 50)

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# --- 1. 保持 BpRNAProcessor 不变 ---
class BpRNAProcessor:
    def __init__(self):
        self.base_map = {
            'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3,
            'N': 4, 'R': 4, 'Y': 4, 'M': 4, 'K': 4, 'S': 4, 'W': 4, 'H': 4, 'B': 4, 'V': 4, 'D': 4
        }
        self.num_bases = 4

    def seq_to_onehot(self, sequence):
        sequence = sequence.upper()
        length = len(sequence)
        one_hot = np.zeros((length, self.num_bases), dtype=np.float32)
        for i, char in enumerate(sequence):
            idx = self.base_map.get(char, 4)
            if idx < 4:
                one_hot[i, idx] = 1.0
        return torch.from_numpy(one_hot)

    def struct_to_matrix(self, structure):
        length = len(structure)
        matrix = np.zeros((length, length), dtype=np.float32)
        stacks = {'(': [], '[': [], '{': [], '<': []}
        pairs_map = {')': '(', ']': '[', '}': '{', '>': '<'}

        for i, char in enumerate(structure):
            if char in stacks:
                stacks[char].append(i)
            elif char in pairs_map:
                open_char = pairs_map[char]
                if len(stacks[open_char]) > 0:
                    j = stacks[open_char].pop()
                    matrix[i, j] = 1.0
                    matrix[j, i] = 1.0
        return torch.from_numpy(matrix)


# --- 2. 修改 Dataset 以支持多文件读取 ---

class MultiFileDataset(Dataset):
    def __init__(self, data_dir, max_len=600):
        self.processor = BpRNAProcessor()
        self.data = []

        file_list = sorted(glob.glob(os.path.join(data_dir, "*.st")))
        print(f"找到 {len(file_list)} 个文件，开始加载并过滤 (MaxLen={max_len})...")

        for fpath in file_list:
            with open(fpath) as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            # 简化的解析逻辑 (假设标准 bpRNA 格式)
            current_entry = {}
            state = 0  # 0:Name, 1:Seq, 2:Struct

            for line in lines:
                if line.startswith("#Name:"):
                    if 'seq' in current_entry and 'struct' in current_entry:
                        self._add_if_valid(current_entry, max_len)
                    current_entry = {}
                    state = 1
                elif state == 1 and not line.startswith("#"):
                    # 简单的启发式：如果是纯字母
                    if all(c.upper() in "ACGUTNRYMKSWHBVD" for c in line):
                        current_entry['seq'] = line
                        state = 2
                elif state == 2:
                    # 简单的启发式：如果是括号点号
                    if any(c in "().[]{}<>" for c in line):
                        current_entry['struct'] = line
                        state = 0

            # 添加最后一个
            if 'seq' in current_entry and 'struct' in current_entry:
                self._add_if_valid(current_entry, max_len)

        print(f"加载完成，有效数据共 {len(self.data)} 条。")

    def _add_if_valid(self, entry, max_len):
        if len(entry['seq']) <= max_len:
            self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        # 截断对齐
        l = min(len(e['seq']), len(e['struct']))
        s_ten = self.processor.seq_to_onehot(e['seq'][:l])
        l_mat = self.processor.struct_to_matrix(e['struct'][:l])
        return s_ten, l_mat


def collate_pad(batch):
    # Pad 到当前 Batch 最大长度
    max_len = max([x[0].shape[0] for x in batch])
    b_size = len(batch)

    seqs = torch.zeros(b_size, max_len, 4)
    labels = torch.zeros(b_size, max_len, max_len)
    masks = torch.zeros(b_size, max_len)  # 1D mask 即可

    for i, (s, l) in enumerate(batch):
        n = s.shape[0]
        seqs[i, :n] = s
        labels[i, :n, :n] = l
        masks[i, :n] = 1.0
    return seqs, labels, masks
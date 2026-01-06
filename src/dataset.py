import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Import Config for shared constants
try:
    from .config import Config
    DEFAULT_MAX_N_RATIO = Config.MAX_N_RATIO
except (ImportError, AttributeError):
    # Fallback if config not available or running as standalone
    DEFAULT_MAX_N_RATIO = 0.2


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

        file_list = sorted(glob.glob(os.path.join(data_dir, "*.dbn")))
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


class MultiFileDatasetUpgrade(Dataset):
    def __init__(self, data_dir_or_file, max_len=600):
        self.processor = BpRNAProcessor()
        self.data = []

        # 1. 获取文件列表
        if os.path.isfile(data_dir_or_file):
            file_list = [data_dir_or_file]
        else:
            file_list = sorted(glob.glob(os.path.join(data_dir_or_file, "*.dbn")))
            # 如果找不到 .dbn，试试 .st (你刚才提到的后缀)
            if not file_list:
                file_list = sorted(glob.glob(os.path.join(data_dir_or_file, "*.st")))

        print(f"🧐 正在扫描 {len(file_list)} 个文件 (MaxLen={max_len})...")

        # 统计计数
        stats = {"total": 0, "kept": 0, "long": 0, "error": 0}

        for fpath in file_list:
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    # 预处理：去掉纯空行
                    lines = [line.strip() for line in f if line.strip()]

                # === 核心解析状态机 ===
                # state 0: 找 Name
                # state 1: 找 Seq (纯字母)
                # state 2: 找 Struct (含括号)

                current_entry = {}
                state = 0

                for line in lines:
                    # 1. 如果遇到 #Name: 或 >，说明是一条新数据的开始
                    if line.startswith("#Name:") or line.startswith(">"):
                        # 如果上一条数据还没存，先存上一条 (如果有的话)
                        if state == 2 and 'seq' in current_entry and 'struct' in current_entry:
                            self._add_if_valid(current_entry, max_len, stats)

                        # 重置状态，开始新的一条
                        current_entry = {}
                        # Extract name from the line
                        if line.startswith("#Name:"):
                            current_entry['name'] = line.split(":", 1)[1].strip()
                        else:  # starts with ">"
                            # Extract name up to first whitespace
                            current_entry['name'] = line[1:].split()[0] if len(line) > 1 else "unknown"
                        state = 1  # 下一步该找 Seq 了
                        continue

                    # 2. 如果是注释行 (#Length, #PageNumber)，直接跳过
                    if line.startswith("#"):
                        continue

                    # 3. 找序列 (State 1)
                    if state == 1:
                        # 启发式判断：如果包含括号，那说明漏掉了 Seq，直接变成 Struct 了 (格式错误)
                        if any(c in "().[]{}<>" for c in line):
                            # 尝试补救：如果是第一行就是结构，那这数据没法要
                            state = 0
                            continue

                        # 正常的序列应该只包含字母
                        # 你的数据里有 'AGAG...'
                        current_entry['seq'] = line.upper().replace('T', 'U')
                        state = 2  # 下一步找 Struct
                        continue

                    # 4. 找结构 (State 2)
                    if state == 2:
                        # 结构行特征：包含括号或点
                        if any(c in "().[]{}<>" for c in line):
                            current_entry['struct'] = line
                            # 找到了完整的一对，尝试保存
                            self._add_if_valid(current_entry, max_len, stats)
                            # 保存完归零，准备找下一个 Name
                            current_entry = {}
                            state = 0
                        else:
                            # 到了 State 2 却没看到括号，可能是多行序列？暂不处理复杂情况
                            state = 0

                # 循环结束，别忘了最后一条
                if 'seq' in current_entry and 'struct' in current_entry:
                    self._add_if_valid(current_entry, max_len, stats)

            except Exception as e:
                print(f"⚠️ 读取 {os.path.basename(fpath)} 失败: {e}")

        print("\n" + "=" * 30)
        print(f"📊 加载报告 (MaxLen={max_len})")
        print(f"✅ 最终入库: {stats['kept']}")
        print(f"❌ 超长丢弃: {stats['long']}")
        print(f"❌ 格式/N多: {stats['error']}")
        print("=" * 30 + "\n")

    def _add_if_valid(self, entry, max_len, stats):
        seq = entry['seq']
        struct = entry['struct']
        stats["total"] += 1

        # 1. 长度检查
        if len(seq) > max_len:
            stats["long"] += 1
            return

        # 2. 长度匹配检查
        if len(seq) != len(struct):
            stats["error"] += 1
            return

        # 3. 内容检查 (允许 20% 的 N，因为预训练不用太严)
        if seq.count('N') / len(seq) > DEFAULT_MAX_N_RATIO:
            stats["error"] += 1
            return

        # 4. 通过
        self.data.append(entry)
        stats["kept"] += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        s_ten = self.processor.seq_to_onehot(e['seq'])
        l_mat = self.processor.struct_to_matrix(e['struct'])
        return s_ten, l_mat
    
    def get_name(self, idx):
        """Get the name/identifier of a sample by index."""
        return self.data[idx].get('name', f'unknown_{idx}')
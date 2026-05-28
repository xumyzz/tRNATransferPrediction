import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import RNA

# Add scripts directory to path for format_utils
_script_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts')
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from format_utils import sniff_format, has_pseudoknot


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


# --- 2. 新增的辅助函数：计算 BPPM ---
def get_bppm_matrix(seq):
    """
    使用 ViennaRNA 计算序列的热力学配对概率矩阵 (BPPM)
    返回 shape 为 (L, L) 的 numpy 数组
    """
    L = len(seq)
    # 如果序列中包含 'N'，RNAfold 可能会报错或表现异常
    # 这里为了安全起见，把 'N' 替换成 'A'（仅用于计算热力学特征，不影响主序列）
    safe_seq = seq.replace('N', 'A') 
    
    fc = RNA.fold_compound(safe_seq)
    
    try:
        fc.pf()  # 计算配分函数
        bpp = fc.bpp()  # 提取配对概率
    except Exception as e:
        print(f"⚠️ RNAfold Error on sequence (Length: {L}): {e}")
        return np.zeros((L, L), dtype=np.float32)
        
    matrix = np.zeros((L, L), dtype=np.float32)
    for i in range(1, L + 1):
        for j in range(i + 1, L + 1):
            prob = bpp[i][j]
            matrix[i-1, j-1] = prob
            matrix[j-1, i-1] = prob # 保持矩阵对称
            
    return matrix


# --- 3. 更新后的 collate_pad ---
def collate_pad(batch):
    """
    Pad 到当前 Batch 最大长度
    batch 是一个 list，每个元素是 (s_ten, bppm_ten, l_mat) 三元组
    """
    max_len = max([x[0].shape[0] for x in batch])
    b_size = len(batch)

    # 初始化存储张量 (使用全 0 进行 Padding)
    seqs = torch.zeros(b_size, max_len, 4)
    bppms = torch.zeros(b_size, max_len, max_len) # 新增: 给 BPPM 准备的存储空间
    labels = torch.zeros(b_size, max_len, max_len)
    masks = torch.zeros(b_size, max_len)  # 1D mask

    # 遍历每个样本并装填
    for i, (s, b, l) in enumerate(batch):
        n = s.shape[0]
        
        # 将真实数据填入对应的左上角区域，多余的部分自然是 0
        seqs[i, :n] = s
        bppms[i, :n, :n] = b  # 装填 BPPM
        labels[i, :n, :n] = l
        masks[i, :n] = 1.0
        
    # 注意返回顺序，要和 train.py 里解包的顺序一致：seqs, bppms, labels, masks
    return seqs, bppms, labels, masks


# --- 4. 数据集类 ---
class MultiFileDatasetUpgrade(Dataset):
    """
    Dataset that supports both .st and .dbn formats with robust parsing.
    Tracks sample names for clustering and export.
    """
    def __init__(self, data_dir_or_file, max_len=600, n_threshold=0.2, allow_pseudoknot=False):
        self.processor = BpRNAProcessor()
        self.data = []
        self.names = []
        self.max_len = max_len
        self.n_threshold = n_threshold
        self.allow_pseudoknot = allow_pseudoknot

        if os.path.isfile(data_dir_or_file):
            file_list = [data_dir_or_file]
        elif os.path.isdir(data_dir_or_file):
            st_files = sorted(glob.glob(os.path.join(data_dir_or_file, "*.st")))
            dbn_files = sorted(glob.glob(os.path.join(data_dir_or_file, "*.dbn")))
            file_list = st_files + dbn_files
        else:
            raise ValueError(f"Invalid path: {data_dir_or_file}")

        print(f"🧐 Scanning {len(file_list)} files (MaxLen={max_len}, AllowPseudoknot={allow_pseudoknot})...")

        stats = {
            "total": 0, "kept": 0, "too_long": 0, "length_mismatch": 0,
            "too_many_n": 0, "invalid_bases": 0, "parse_error": 0,
            "pseudoknot_filtered": 0, "sniffed_st_in_dbn": 0, "unknown_format_files": 0
        }

        for fpath in file_list:
            try:
                sniffed = sniff_format(fpath) 
                if sniffed is None:
                    stats["unknown_format_files"] += 1
                    continue
                if fpath.endswith('.dbn') and sniffed == 'st':
                    stats["sniffed_st_in_dbn"] += 1
                
                if sniffed == 'st':
                    self._parse_st_file(fpath, stats)
                elif sniffed == 'dbn':
                    self._parse_dbn_file(fpath, stats)
            except Exception as e:
                stats["parse_error"] += 1

        print("\n" + "=" * 50)
        print(f"📊 Loading Report (MaxLen={max_len})")
        print(f"✅ Total kept: {stats['kept']}")
        print(f"❌ Too long: {stats['too_long']}")
        print(f"❌ Length mismatch: {stats['length_mismatch']}")
        print(f"❌ Too many Ns: {stats['too_many_n']}")
        print(f"❌ Invalid bases: {stats['invalid_bases']}")
        print(f"❌ Pseudoknot filtered: {stats['pseudoknot_filtered']}")
        print(f"❌ Parse errors: {stats['parse_error']}")
        print("=" * 50 + "\n")

    def _parse_st_file(self, fpath, stats):
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.rstrip() for line in f]
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("#Name:"):
                name = line[6:].strip()
                i += 1
                while i < len(lines) and lines[i].strip().startswith("#"): i += 1
                if i >= len(lines): break
                seq_line = lines[i].strip()
                i += 1
                if i >= len(lines): break
                struct_line = lines[i].strip()
                self._add_if_valid({
                    'name': name,
                    'seq': seq_line.upper().replace('T', 'U'),
                    'struct': struct_line
                }, stats)
            i += 1

    def _parse_dbn_file(self, fpath, stats):
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.rstrip() for line in f]
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if line.startswith(">"):
                name = line[1:].strip()
                if not name: name = f"seq_{i+1}"
                i += 1
                seq_line = ""
                while i < len(lines):
                    line = lines[i].strip()
                    if line:
                        seq_line = line
                        break
                    i += 1
                if not seq_line: break
                i += 1
                struct_line = ""
                while i < len(lines):
                    line = lines[i].strip()
                    if line:
                        struct_line = line
                        break
                    i += 1
                if not struct_line: break
                self._add_if_valid({
                    'name': name,
                    'seq': seq_line.upper().replace('T', 'U'),
                    'struct': struct_line
                }, stats)
            i += 1

    def _add_if_valid(self, entry, stats):
        seq = entry['seq']
        struct = entry['struct']
        name = entry.get('name', f'unknown_{stats["total"]}')
        stats["total"] += 1
        
        if len(seq) > self.max_len: stats["too_long"] += 1; return
        if len(seq) != len(struct): stats["length_mismatch"] += 1; return
        valid_bases = set('ACGUN')
        if not all(c in valid_bases for c in seq): stats["invalid_bases"] += 1; return
        if len(seq) > 0 and seq.count('N') / len(seq) > self.n_threshold: stats["too_many_n"] += 1; return
        if not self.allow_pseudoknot and has_pseudoknot(struct): stats["pseudoknot_filtered"] += 1; return
        
        self.data.append({'seq': seq, 'struct': struct})
        self.names.append(name)
        stats["kept"] += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        
        # 恢复了安全的截断对齐逻辑
        l = min(len(e['seq']), len(e['struct']))
        seq = e['seq'][:l]
        struct = e['struct'][:l]
        
        # 1. 序列转换
        s_ten = self.processor.seq_to_onehot(seq)
        
        # 2. 标签转换
        l_mat = self.processor.struct_to_matrix(struct)
        
        # 3. 计算 BPPM 矩阵
        bppm_np = get_bppm_matrix(seq)
        bppm_ten = torch.tensor(bppm_np, dtype=torch.float32)
        
        # 返回三元组
        return s_ten, bppm_ten, l_mat
    
    def get_name(self, idx):
        return self.names[idx]
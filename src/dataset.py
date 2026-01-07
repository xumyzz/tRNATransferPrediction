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
    """
    Dataset that supports both .st and .dbn formats with robust parsing.
    Tracks sample names for clustering and export.
    
    Formats supported:
    - .st: bpRNA format with #Name: header
    - .dbn: FASTA-like format with > header, sequence, dot-bracket structure
    """
    def __init__(self, data_dir_or_file, max_len=600, n_threshold=0.2):
        self.processor = BpRNAProcessor()
        self.data = []
        self.names = []  # Track names for each sample
        self.max_len = max_len
        self.n_threshold = n_threshold

        # 1. Get file list
        if os.path.isfile(data_dir_or_file):
            file_list = [data_dir_or_file]
        elif os.path.isdir(data_dir_or_file):
            # Support both .st and .dbn files
            st_files = sorted(glob.glob(os.path.join(data_dir_or_file, "*.st")))
            dbn_files = sorted(glob.glob(os.path.join(data_dir_or_file, "*.dbn")))
            file_list = st_files + dbn_files
        else:
            raise ValueError(f"Invalid path: {data_dir_or_file} (path does not exist or is not a file or directory)")

        print(f"🧐 Scanning {len(file_list)} files (MaxLen={max_len})...")

        # Statistics
        stats = {
            "total": 0, 
            "kept": 0, 
            "too_long": 0, 
            "length_mismatch": 0,
            "too_many_n": 0,
            "invalid_bases": 0,
            "parse_error": 0
        }

        for fpath in file_list:
            try:
                # Determine format by extension
                if fpath.endswith('.st'):
                    self._parse_st_file(fpath, stats)
                elif fpath.endswith('.dbn'):
                    self._parse_dbn_file(fpath, stats)
            except Exception as e:
                print(f"⚠️ Error reading {os.path.basename(fpath)}: {e}")
                stats["parse_error"] += 1

        print("\n" + "=" * 50)
        print(f"📊 Loading Report (MaxLen={max_len})")
        print(f"✅ Total kept: {stats['kept']}")
        print(f"❌ Too long: {stats['too_long']}")
        print(f"❌ Length mismatch: {stats['length_mismatch']}")
        print(f"❌ Too many Ns: {stats['too_many_n']}")
        print(f"❌ Invalid bases: {stats['invalid_bases']}")
        print(f"❌ Parse errors: {stats['parse_error']}")
        print("=" * 50 + "\n")

    def _parse_st_file(self, fpath, stats):
        """Parse bpRNA .st format files"""
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.rstrip() for line in f]
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for #Name: header
            if line.startswith("#Name:"):
                name = line[6:].strip()
                
                # Skip other comment lines
                i += 1
                while i < len(lines) and lines[i].strip().startswith("#"):
                    i += 1
                
                # Next should be sequence
                if i >= len(lines):
                    break
                seq_line = lines[i].strip()
                
                # Next should be structure
                i += 1
                if i >= len(lines):
                    break
                struct_line = lines[i].strip()
                
                # Validate and add
                self._add_if_valid({
                    'name': name,
                    'seq': seq_line.upper().replace('T', 'U'),
                    'struct': struct_line
                }, stats)
            i += 1

    def _parse_dbn_file(self, fpath, stats):
        """Parse .dbn format files (FASTA-like with dot-bracket)"""
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.rstrip() for line in f]
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Look for header line starting with >
            if line.startswith(">"):
                name = line[1:].strip()
                if not name:
                    # Use a default name with line number
                    name = f"seq_{i+1}"
                
                # Next non-empty line should be sequence
                i += 1
                seq_line = ""
                while i < len(lines):
                    line = lines[i].strip()
                    if line:
                        seq_line = line
                        break
                    i += 1
                
                if not seq_line:
                    break
                
                # Next non-empty line should be structure
                i += 1
                struct_line = ""
                while i < len(lines):
                    line = lines[i].strip()
                    if line:
                        struct_line = line
                        break
                    i += 1
                
                if not struct_line:
                    break
                
                # Validate and add
                self._add_if_valid({
                    'name': name,
                    'seq': seq_line.upper().replace('T', 'U'),
                    'struct': struct_line
                }, stats)
            i += 1

    def _add_if_valid(self, entry, stats):
        """Validate entry and add if it passes all checks"""
        seq = entry['seq']
        struct = entry['struct']
        name = entry.get('name', f'unknown_{stats["total"]}')
        stats["total"] += 1

        # 1. Length check
        if len(seq) > self.max_len:
            stats["too_long"] += 1
            return

        # 2. Length match check
        if len(seq) != len(struct):
            stats["length_mismatch"] += 1
            return

        # 3. Validate sequence only contains valid bases (A, C, G, U, N)
        valid_bases = set('ACGUN')
        if not all(c in valid_bases for c in seq):
            stats["invalid_bases"] += 1
            return

        # 4. N threshold check
        if len(seq) > 0 and seq.count('N') / len(seq) > self.n_threshold:
            stats["too_many_n"] += 1
            return

        # 5. All checks passed - add to dataset
        self.data.append({'seq': seq, 'struct': struct})
        self.names.append(name)
        stats["kept"] += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        s_ten = self.processor.seq_to_onehot(e['seq'])
        l_mat = self.processor.struct_to_matrix(e['struct'])
        return s_ten, l_mat
    
    def get_name(self, idx):
        """Get the name of a sample by index"""
        return self.names[idx]
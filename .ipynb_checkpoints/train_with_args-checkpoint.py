"""
伪 tRNA-miRNA 嵌合体数据生成器
================================
Generates pseudo tRNA-miRNA chimeric RNA sequences with ground truth contact maps
for training deep learning models for RNA secondary structure prediction.

原理:
  嵌合体 = 5' tRNA片段 + 中间插入片段(miRNA前体) + 3' tRNA片段
  - tRNA 的 5' 和 3' 片段来自同一条 tRNA，保留其跨域长程配对（acceptor stem）
  - 中间插入片段来自 bpRNA 中非 tRNA 家族的 RNA，使用其已知二级结构
  - 跨域（tRNA↔insert）位置 contact = 0，训练时通过 valid_mask 排除

用法:
  python pseudo_chimera_generator.py \
      --bprna_file /path/to/bpRNA.txt \
      --output ./pseudo_chimeras_train.pkl \
      --num_samples 8000 --seed 42

  # 如果插入片段无结构信息，可用 ViennaRNA 在线预测:
  python pseudo_chimera_generator.py \
      --bprna_file /path/to/bpRNA.txt \
      --output ./pseudo_chimeras_train.pkl \
      --use_vienna --vienna_path RNAfold
"""

import numpy as np
import pickle
import random
import os
import sys
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class RNAEntry:
    """单条 RNA 数据"""
    name: str
    sequence: str          # AUGC, T 已转为 U
    structure: str         # dot-bracket 格式
    length: int = 0

    def __post_init__(self):
        self.length = len(self.sequence)


@dataclass
class ChimeraSample:
    """单条伪嵌合体"""
    sequence: str                          # 完整序列
    contact_map: np.ndarray                # (L, L) float32, 对称, 二值
    domain_labels: np.ndarray              # (L,) int32, 0=tRNA, 2=miRNA
    # 元信息
    trna_seq_5: str
    trna_seq_3: str
    insert_seq: str
    trna_name: str
    insert_name: str
    lengths: Dict[str, int] = field(default_factory=dict)


# ============================================================================
# Dot-Bracket 解析
# ============================================================================

# 支持的碱基配对括号
PAIR_BRACKETS = {
    '(': ')', '[': ']', '{': '}', '<': '>',
}
OPEN_BRACKETS = set(PAIR_BRACKETS.keys())
CLOSE_TO_OPEN = {v: k for k, v in PAIR_BRACKETS.items()}

# RNA 碱基互补规则（包含 GU wobble）
CANONICAL_PAIRS = {
    ('A', 'U'), ('U', 'A'),
    ('C', 'G'), ('G', 'C'),
    ('G', 'U'), ('U', 'G'),
}


def dotbracket_to_pairs(dotbracket: str) -> List[Tuple[int, int]]:
    """
    将 dot-bracket 表示转换为碱基配对列表。
    支持伪结扩展标记: [ ] { } < >
    
    Args:
        dotbracket: dot-bracket 格式的结构字符串
    
    Returns:
        [(i, j), ...] 配对位置列表 (从0开始)
    """
    stacks = {bracket: [] for bracket in OPEN_BRACKETS}
    pairs = []
    
    for i, c in enumerate(dotbracket):
        if c in OPEN_BRACKETS:
            stacks[c].append(i)
        elif c in CLOSE_TO_OPEN:
            opener = CLOSE_TO_OPEN[c]
            if stacks[opener]:
                j = stacks[opener].pop()
                pairs.append((j, i))
    
    return pairs


def pairs_to_contact_map(length: int, pairs: List[Tuple[int, int]]) -> np.ndarray:
    """将配对列表转换为 (L, L) 对称二值接触图"""
    cmap = np.zeros((length, length), dtype=np.float32)
    for i, j in pairs:
        if 0 <= i < length and 0 <= j < length:
            cmap[i, j] = 1.0
            cmap[j, i] = 1.0
    return cmap


def contact_map_to_pairs(cmap: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int]]:
    """将 contact map 转回配对列表（上三角，去重）"""
    L = cmap.shape[0]
    indices = np.where(np.triu(cmap, k=1) >= threshold)
    return list(zip(indices[0].tolist(), indices[1].tolist()))


# ============================================================================
# bpRNA 文件解析
# ============================================================================

def parse_dbn_file(filepath: str, max_entries: int = None) -> List[RNAEntry]:
    """
    解析单个 .dbn 文件（bpRNA 单条目格式）。
    
    .dbn 格式:
        #Name: bpRNA_CRW_1
        #Length: 1434
        #PageNumber: 2
        <SEQUENCE>
        <STRUCTURE>
    
    Args:
        filepath: .dbn 文件路径
        max_entries: 未使用（单个文件只有一条），保留接口一致性
    
    Returns:
        包含 0 或 1 个 RNAEntry 的列表
    """
    entries = []
    name = os.path.splitext(os.path.basename(filepath))[0]
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 跳过注释行（以 # 开头），收集序列行和结构行
    seq_candidates = []
    for line in lines:
        stripped = line.rstrip('\n').rstrip('\r').strip()
        if not stripped or stripped.startswith('#'):
            continue
        seq_candidates.append(stripped)
    
    # .dbn 格式: 序列行 + 结构行（最后两行）
    # 也可能有多条结构（如多个 page），我们按序列行在前、结构行在后的规则取
    if len(seq_candidates) >= 2:
        # 结构行通常包含 (, ), . 等字符，而序列行是纯碱基
        # 取不含结构字符的行为序列，含结构字符的行为结构
        struct_chars = set('().[]{}<>_')
        seq_lines = []
        struct_lines = []
        for s in seq_candidates:
            if any(c in struct_chars for c in s) or s.startswith('.'):
                struct_lines.append(s)
            else:
                seq_lines.append(s)
        
        if seq_lines and struct_lines:
            sequence = seq_lines[0].upper().replace('T', 'U')
            structure = struct_lines[0]
        elif len(seq_candidates) >= 2:
            # 回退：倒数第2行是序列，倒数第1行是结构
            sequence = seq_candidates[-2].upper().replace('T', 'U')
            structure = seq_candidates[-1]
        else:
            return entries
    else:
        return entries
    
    # 验证
    valid_bases = set('AUGC')
    if len(sequence) == 0:
        return entries
    if any(b not in valid_bases for b in sequence):
        return entries
    
    if len(structure) == 0 or len(structure) != len(sequence):
        structure = '.' * len(sequence)
    
    entries.append(RNAEntry(
        name=name,
        sequence=sequence,
        structure=structure,
    ))
    
    return entries


def parse_dbn_dir(dirpath: str, max_entries: int = None) -> List[RNAEntry]:
    """
    解析包含多个 .dbn 文件的目录。
    
    Args:
        dirpath: 包含 .dbn 文件的目录路径
        max_entries: 最大读取条目数（用于测试）
    
    Returns:
        RNAEntry 列表
    """
    entries = []
    skipped = 0
    
    # 收集所有 .dbn 文件
    dbn_files = sorted([
        os.path.join(dirpath, f) for f in os.listdir(dirpath)
        if f.endswith('.dbn')
    ])
    
    if not dbn_files:
        print(f"[parse_dbn_dir] 目录中无 .dbn 文件: {dirpath}")
        return entries
    
    print(f"[parse_dbn_dir] 发现 {len(dbn_files)} 个 .dbn 文件")
    
    for fi, filepath in enumerate(dbn_files):
        if max_entries and len(entries) >= max_entries:
            break
        
        result = parse_dbn_file(filepath)
        if result:
            entries.extend(result)
        else:
            skipped += 1
        
        if fi % 1000 == 0 and fi > 0:
            print(f"  已处理 {fi}/{len(dbn_files)} 个文件...")
    
    print(f"[parse_dbn_dir] 加载 {len(entries)} 条, 跳过 {skipped} 条")
    return entries


def parse_bprna_txt_file(filepath: str, max_entries: int = None) -> List[RNAEntry]:
    """
    解析 bpRNA 格式的单个文本文件（FASTA 风格，多条条目）。
    
    bpRNA 标准格式:
        >name
        SEQUENCE
        STRUCTURE
    
    Args:
        filepath: 文本文件路径
        max_entries: 最大读取条目数
    
    Returns:
        RNAEntry 列表
    """
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    skipped = 0
    while i < len(lines):
        line = lines[i].rstrip('\n').rstrip('\r')
        if line.startswith('>'):
            name = line[1:].strip()
            sequence = ''
            structure = ''
            
            if i + 1 < len(lines):
                sequence = lines[i + 1].rstrip('\n').rstrip('\r').strip()
            if i + 2 < len(lines):
                structure = lines[i + 2].rstrip('\n').rstrip('\r').strip()
            
            # 清洗
            sequence = sequence.upper().replace('T', 'U')
            
            # 验证
            if len(sequence) == 0:
                skipped += 1
                i += 1
                continue
            if len(structure) == 0 or len(structure) != len(sequence):
                structure = '.' * len(sequence)
            
            # 只保留标准碱基
            valid_bases = set('AUGC')
            if any(b not in valid_bases for b in sequence):
                skipped += 1
                i += 3
                continue
            
            entries.append(RNAEntry(
                name=name,
                sequence=sequence,
                structure=structure,
            ))
            i += 3
            
            if max_entries and len(entries) >= max_entries:
                break
        else:
            i += 1
    
    print(f"[parse_bprna_txt] 加载 {len(entries)} 条, 跳过 {skipped} 条")
    return entries


def parse_bprna_file(filepath: str, max_entries: int = None) -> List[RNAEntry]:
    """
    智能解析 bpRNA 数据：自动识别目录或单文件格式。
    
    支持三种输入:
      - 目录:     包含多个 .dbn 文件  → parse_dbn_dir()
      - 单文件 .dbn:  单条 .dbn 格式    → parse_dbn_file()
      - 单文件 .txt:  bpRNA FASTA 格式   → parse_bprna_txt_file()
    
    Args:
        filepath: 目录路径或文件路径
        max_entries: 最大读取条目数
    
    Returns:
        RNAEntry 列表
    """
    if os.path.isdir(filepath):
        return parse_dbn_dir(filepath, max_entries)
    elif filepath.endswith('.dbn'):
        return parse_dbn_file(filepath, max_entries)
    else:
        return parse_bprna_txt_file(filepath, max_entries)


# ============================================================================
# tRNA 识别与分类
# ============================================================================

# 用于识别 tRNA 的关键词（不区分大小写）
TRNA_KEYWORDS = [
    'trna', 't_rna', 'transfer_rna', 'transfer rna',
    '_TR',           # bpRNA 家族标记: RFAM_TR 等
]

# tRNA 序列长度典型范围
TRNA_LEN_MIN = 60
TRNA_LEN_MAX = 95


def is_trna_entry(entry: RNAEntry,
                  keywords: List[str] = None,
                  len_range: Tuple[int, int] = None) -> bool:
    """
    判断一条 RNA 是否为 tRNA。
    判断依据: 名称关键词匹配 OR 长度在 tRNA 典型范围内
    
    Args:
        entry: RNA 条目
        keywords: 自定义关键词列表
        len_range: 自定义长度范围 (min, max)
    
    Returns:
        True 如果判定为 tRNA
    """
    if keywords is None:
        keywords = TRNA_KEYWORDS
    if len_range is None:
        len_range = (TRNA_LEN_MIN, TRNA_LEN_MAX)
    
    name_lower = entry.name.lower()
    
    # 方法1: 名称匹配
    for kw in keywords:
        if kw.lower() in name_lower:
            return True
    
    # 方法2: 长度范围（辅助判断，不是唯一依据）
    # 注意: 仅长度判断可能误判，因此需要配合其他特征
    # 此处仅用名称，用户可通过 --tRNA_family_keyword 自定义
    
    return False


def classify_entries(entries: List[RNAEntry],
                     trna_keyword: str = 'tRNA') -> Tuple[List[RNAEntry], List[RNAEntry]]:
    """
    将条目分为 tRNA 和非 tRNA 两组
    
    Args:
        entries: RNA 条目列表
        trna_keyword: tRNA 识别关键词
    
    Returns:
        (tRNA列表, 非tRNA列表)
    """
    trna_list = []
    non_trna_list = []
    
    # 允许多关键词，用逗号分隔
    keywords = [kw.strip() for kw in trna_keyword.split(',')]
    
    for entry in entries:
        if is_trna_entry(entry, keywords=keywords):
            trna_list.append(entry)
        else:
            non_trna_list.append(entry)
    
    # 去重: 按序列去重（保留第一个出现的）
    trna_list = _deduplicate_by_sequence(trna_list)
    non_trna_list = _deduplicate_by_sequence(non_trna_list)
    
    print(f"[classify] tRNA: {len(trna_list)}, 非tRNA: {len(non_trna_list)}")
    return trna_list, non_trna_list


def _deduplicate_by_sequence(entries: List[RNAEntry]) -> List[RNAEntry]:
    """按序列去重，保留第一个"""
    seen = set()
    result = []
    for entry in entries:
        if entry.sequence not in seen:
            seen.add(entry.sequence)
            result.append(entry)
    return result


# ============================================================================
# 核心: 伪嵌合体生成
# ============================================================================

class PseudoChimeraGenerator:
    """
    伪嵌合体生成器。
    
    从 tRNA + 非tRNA RNA 数据中生成合成嵌合体序列和 contact map。
    
    tRNA 拆分策略:
        5' 片段: tRNA[:split_5]   — 包含 acceptor stem 5' 端, D-stem, anticodon stem
        3' 片段: tRNA[-split_3:]  — 包含 T-stem 3' 端, acceptor stem 3' 端
        丢弃:    tRNA[split_5:-split_3] — T-loop 等中间区域
    
    这种拆分确保 5'↔3' 的 acceptor stem 配对能被正确捕获为
    跨长距离（≈插入片段长度）的长程配对。
    """
    
    def __init__(
        self,
        trna_entries: List[RNAEntry],
        insert_entries: List[RNAEntry],
        trna_5prime_len: int = 43,
        trna_3prime_len: int = 15,
        insert_len_min: int = 100,
        insert_len_max: int = 150,
        seed: int = 42,
        use_vienna: bool = False,
        vienna_path: str = 'RNAfold',
    ):
        """
        Args:
            trna_entries: tRNA 数据
            insert_entries: 非tRNA 插入片段数据
            trna_5prime_len: 5' tRNA 片段目标长度
            trna_3prime_len: 3' tRNA 片段目标长度
            insert_len_min: 插入片段最小长度
            insert_len_max: 插入片段最大长度
            seed: 随机种子
            use_vienna: 是否使用 ViennaRNA 预测无结构 insert 的结构
            vienna_path: RNAfold 可执行文件路径
        """
        self.trna_entries = trna_entries
        self.insert_entries = insert_entries
        self.trna_5prime_len = trna_5prime_len
        self.trna_3prime_len = trna_3prime_len
        self.insert_len_min = insert_len_min
        self.insert_len_max = insert_len_max
        self.use_vienna = use_vienna
        self.vienna_path = vienna_path
        
        random.seed(seed)
        np.random.seed(seed)
        
        # 预构建 tRNA 索引: 按长度分桶
        self._trna_by_len = self._bucket_by_length(trna_entries)
        self._insert_by_len = self._bucket_by_length(insert_entries)
        
        # 统计
        self._stats = {
            'generated': 0,
            'skipped_short_trna': 0,
            'skipped_short_insert': 0,
            'skipped_no_pairs': 0,
        }
    
    def _bucket_by_length(self, entries: List[RNAEntry]) -> Dict[str, List[int]]:
        """按长度分桶，返回 {len_range: [indices]}"""
        buckets = defaultdict(list)
        bucket_boundaries = [0, 50, 80, 100, 120, 150, 200, 300, 500, 1000, 10000]
        
        for idx, entry in enumerate(entries):
            for i in range(len(bucket_boundaries) - 1):
                if bucket_boundaries[i] <= entry.length < bucket_boundaries[i + 1]:
                    key = f"{bucket_boundaries[i]}-{bucket_boundaries[i + 1]}"
                    buckets[key].append(idx)
                    break
        
        return dict(buckets)
    
    def _pick_random_entry(self,
                           entries: List[RNAEntry],
                           buckets: Dict[str, List[int]],
                           min_len: int,
                           max_len: int) -> Optional[int]:
        """根据长度范围随机选择一条 entry 的索引"""
        candidate_indices = []
        for key, indices in buckets.items():
            lo, hi = key.split('-')
            lo, hi = int(lo), int(hi)
            # 有交集
            if lo < max_len and hi > min_len:
                candidate_indices.extend(indices)
        
        if not candidate_indices:
            # 回退: 遍历全部
            candidate_indices = [
                i for i, e in enumerate(entries)
                if min_len <= e.length <= max_len
            ]
        
        if not candidate_indices:
            return None
        
        return random.choice(candidate_indices)
    
    def _extract_trna_fragments(
        self, trna: RNAEntry
    ) -> Optional[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
        """
        从 tRNA 中提取 5' 和 3' 片段及其 contact map 部分。
        
        Returns:
            (trna_seq_5, trna_seq_3, cmap_5, cmap_3, cmap_cross)
            或 None（如果 tRNA 太短）
        """
        L = trna.length
        min_required = self.trna_5prime_len + self.trna_3prime_len + 2
        
        if L < min_required:
            self._stats['skipped_short_trna'] += 1
            return None
        
        # 提取序列片段
        trna_seq_5 = trna.sequence[:self.trna_5prime_len]
        trna_seq_3 = trna.sequence[-self.trna_3prime_len:]
        
        L5 = len(trna_seq_5)
        L3 = len(trna_seq_3)
        
        # 解析完整 tRNA 的 contact map
        pairs = dotbracket_to_pairs(trna.structure)
        full_cmap = pairs_to_contact_map(L, pairs)
        
        # 拆分 contact map
        # 5' 片段内部配对
        cmap_5 = full_cmap[:L5, :L5].copy()
        # 3' 片段内部配对 (取最后 L3 个位置)
        cmap_3 = full_cmap[-L3:, -L3:].copy()
        # 跨域配对: 5'(前 L5) ↔ 3'(后 L3)
        cmap_cross = full_cmap[:L5, -L3:].copy()
        
        return trna_seq_5, trna_seq_3, cmap_5, cmap_3, cmap_cross
    
    def _extract_insert_fragment(
        self, entry: RNAEntry
    ) -> Optional[Tuple[str, np.ndarray]]:
        """
        从非tRNA条目中提取插入片段及其 contact map。
        
        Returns:
            (insert_seq, cmap_insert) 或 None
        """
        L = entry.length
        target_len = random.randint(self.insert_len_min, self.insert_len_max)
        
        if L < target_len:
            self._stats['skipped_short_insert'] += 1
            return None
        
        # 随机截取
        if L > target_len:
            start = random.randint(0, L - target_len)
        else:
            start = 0
            target_len = L
        
        insert_seq = entry.sequence[start:start + target_len]
        insert_struct = entry.structure[start:start + target_len]
        
        # 解析 contact map
        if insert_struct:
            pairs = dotbracket_to_pairs(insert_struct)
            cmap_insert = pairs_to_contact_map(target_len, pairs)
        elif self.use_vienna:
            cmap_insert = self._predict_vienna(insert_seq)
        else:
            # 无结构信息: 用空图
            cmap_insert = np.zeros((target_len, target_len), dtype=np.float32)
        
        return insert_seq, cmap_insert
    
    def _predict_vienna(self, sequence: str) -> np.ndarray:
        """使用 ViennaRNA RNAfold 预测二级结构"""
        import subprocess
        import tempfile
        
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.fa', delete=False, encoding='utf-8'
            ) as f:
                f.write(f'>temp\n{sequence}\n')
                tmpfile = f.name
            
            result = subprocess.run(
                [self.vienna_path, '--noPS', tmpfile],
                capture_output=True, text=True, timeout=30,
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if sys.platform == 'win32' else 0),
            )
            
            lines = result.stdout.strip().split('\n')
            struct_line = ''
            
            if len(lines) >= 3:
                # 标准 Vienna 输出: 3行
                # >name
                # SEQUENCE
                # ..........( ( (....) ) ) (-12.30)
                struct_line = lines[2].split()[0] if len(lines[2].split()) > 0 else ''
            elif len(lines) >= 2:
                struct_line = lines[1].split()[0] if len(lines[1].split()) > 0 else ''
            elif len(lines) == 1:
                parts = lines[0].split()
                struct_line = parts[1] if len(parts) > 1 else ''
            
            if struct_line and len(struct_line) == len(sequence):
                pairs = dotbracket_to_pairs(struct_line)
                return pairs_to_contact_map(len(sequence), pairs)
            else:
                return np.zeros((len(sequence), len(sequence)), dtype=np.float32)
        
        except FileNotFoundError:
            print("[警告] ViennaRNA (RNAfold) 未找到, 将使用空结构")
            return np.zeros((len(sequence), len(sequence)), dtype=np.float32)
        except Exception as e:
            print(f"[警告] ViennaRNA 预测失败: {e}")
            return np.zeros((len(sequence), len(sequence)), dtype=np.float32)
        finally:
            try:
                os.unlink(tmpfile)
            except Exception:
                pass
    
    def generate_one(self) -> Optional[ChimeraSample]:
        """
        生成一条伪嵌合体。
        
        Returns:
            ChimeraSample 或 None（如果本次生成失败）
        """
        # 1. 随机选择 tRNA
        trna_idx = random.randrange(len(self.trna_entries))
        trna = self.trna_entries[trna_idx]
        
        # 2. 提取 tRNA 片段
        fragments = self._extract_trna_fragments(trna)
        if fragments is None:
            return None
        
        trna_seq_5, trna_seq_3, cmap_5, cmap_3, cmap_cross = fragments
        L5 = len(trna_seq_5)
        L3 = len(trna_seq_3)
        
        # 3. 随机选择插入片段
        insert_idx = random.randrange(len(self.insert_entries))
        insert = self.insert_entries[insert_idx]
        
        result = self._extract_insert_fragment(insert)
        if result is None:
            return None
        
        insert_seq, cmap_insert = result
        insert_len = len(insert_seq)
        
        # 4. 组装完整嵌合体
        full_seq = trna_seq_5 + insert_seq + trna_seq_3
        total_len = L5 + insert_len + L3
        
        # 构建完整 contact map
        full_cmap = np.zeros((total_len, total_len), dtype=np.float32)
        
        # 区域索引
        off_5 = 0
        off_ins = L5
        off_3 = L5 + insert_len
        
        # 5' tRNA 内部
        full_cmap[off_5:off_5+L5, off_5:off_5+L5] = cmap_5
        # Insert 内部
        full_cmap[off_ins:off_ins+insert_len, off_ins:off_ins+insert_len] = cmap_insert
        # 3' tRNA 内部
        full_cmap[off_3:off_3+L3, off_3:off_3+L3] = cmap_3
        # 5' ↔ 3' 跨域长程配对
        full_cmap[off_5:off_5+L5, off_3:off_3+L3] = cmap_cross
        full_cmap[off_3:off_3+L3, off_5:off_5+L5] = cmap_cross.T
        
        # 跨域(tRNA↔insert)保持为 0
        
        # 5. 质量检查
        num_pairs = full_cmap.sum() / 2
        if num_pairs < 2:
            self._stats['skipped_no_pairs'] += 1
            return None
        
        # 6. 域标签
        domain_labels = np.zeros(total_len, dtype=np.int32)
        domain_labels[off_ins:off_ins+insert_len] = 2
        
        self._stats['generated'] += 1
        
        return ChimeraSample(
            sequence=full_seq,
            contact_map=full_cmap,
            domain_labels=domain_labels,
            trna_seq_5=trna_seq_5,
            trna_seq_3=trna_seq_3,
            insert_seq=insert_seq,
            trna_name=trna.name,
            insert_name=insert.name,
            lengths={
                'total': total_len,
                'trna_5': L5,
                'insert': insert_len,
                'trna_3': L3,
            },
        )
    
    def generate(self, num_samples: int, verbose: bool = True) -> List[ChimeraSample]:
        """
        批量生成伪嵌合体。
        
        Args:
            num_samples: 目标样本数
            verbose: 是否打印进度
        
        Returns:
            伪嵌合体样本列表
        """
        chimeras = []
        max_attempts = num_samples * 20  # 允许 20x 的尝试量
        attempts = 0
        
        if verbose:
            print(f"[generate] 目标: {num_samples} 条, 开始生成...")
        
        while len(chimeras) < num_samples and attempts < max_attempts:
            attempts += 1
            sample = self.generate_one()
            if sample is not None:
                chimeras.append(sample)
            
            if verbose and len(chimeras) % 1000 == 0 and len(chimeras) > 0:
                print(f"  已生成 {len(chimeras)}/{num_samples} "
                      f"(尝试 {attempts}, 成功率 {len(chimeras)/attempts:.2%})")
        
        if verbose:
            print(f"[generate] 完成: {len(chimeras)} 条 "
                  f"(尝试 {attempts} 次, 成功率 {len(chimeras)/max(attempts,1):.2%})")
            self.print_stats()
        
        return chimeras
    
    def print_stats(self):
        """打印生成统计"""
        print("\n" + "=" * 50)
        print("生成统计")
        print("=" * 50)
        for key, val in self._stats.items():
            print(f"  {key}: {val}")
        print("=" * 50)


# ============================================================================
# 数据增强: 序列扰动 + ViennaRNA 重折叠
# ============================================================================

def augment_insert_with_mutations(
    insert_seq: str,
    num_mutations: int = 3,
    mutation_types: List[str] = None,
    vienna_path: str = 'RNAfold',
) -> List[Tuple[str, np.ndarray]]:
    """
    对插入片段施加随机突变，并用 ViennaRNA 重新预测结构。
    生成增强变体以增加数据多样性。
    
    Args:
        insert_seq: 原始插入序列
        num_mutations: 突变数量
        mutation_types: ['substitute', 'insert', 'delete'] 的子集
        vienna_path: RNAfold 路径
    
    Returns:
        [(mutated_seq, predicted_cmap), ...]
    """
    import subprocess
    import tempfile
    
    if mutation_types is None:
        mutation_types = ['substitute']
    
    bases = ['A', 'U', 'G', 'C']
    variants = []
    
    for _ in range(min(5, num_mutations * 2)):  # 最多生成 5 个变体
        seq_list = list(insert_seq)
        
        for _ in range(num_mutations):
            mt = random.choice(mutation_types)
            pos = random.randint(0, len(seq_list) - 1)
            
            if mt == 'substitute':
                original = seq_list[pos]
                candidates = [b for b in bases if b != original]
                seq_list[pos] = random.choice(candidates)
            elif mt == 'insert' and len(seq_list) < 200:
                seq_list.insert(pos, random.choice(bases))
            elif mt == 'delete' and len(seq_list) > 20:
                seq_list.pop(pos)
        
        mutated_seq = ''.join(seq_list)
        
        # 用 ViennaRNA 预测新结构
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.fa', delete=False, encoding='utf-8'
            ) as f:
                f.write(f'>temp\n{mutated_seq}\n')
                tmpfile = f.name
            
            result = subprocess.run(
                [vienna_path, '--noPS', tmpfile],
                capture_output=True, text=True, timeout=30,
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if sys.platform == 'win32' else 0),
            )
            
            lines = result.stdout.strip().split('\n')
            struct_line = ''
            if len(lines) >= 3:
                struct_line = lines[2].split()[0] if lines[2].split() else ''
            elif len(lines) >= 2:
                parts = lines[1].split()
                struct_line = parts[1] if len(parts) > 1 else ''
            
            if struct_line and len(struct_line) == len(mutated_seq):
                pairs = dotbracket_to_pairs(struct_line)
                cmap = pairs_to_contact_map(len(mutated_seq), pairs)
                variants.append((mutated_seq, cmap))
        except Exception:
            pass
        finally:
            try:
                os.unlink(tmpfile)
            except Exception:
                pass
    
    return variants


# ============================================================================
# PyTorch Dataset 包装
# ============================================================================

class ChimeraDataset:
    """
    PyTorch-style Dataset for pseudo-chimera data.
    
    输出格式:
        {
            'sequence': str,
            'contact_map': np.ndarray (L, L),
            'domain_labels': np.ndarray (L,),
            'onehot': np.ndarray (L, 4),       # 可选, 一键编码
            'valid_mask': np.ndarray (L, L),   # 有效位置 mask
        }
    
    用法:
        dataset = ChimeraDataset(chimeras, onehot=True, max_len=200)
        dataloader = DataLoader(dataset, batch_size=16, collate_fn=dataset.collate_fn)
    """
    
    # 碱基 → 索引映射
    BASE_TO_IDX = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
    
    def __init__(
        self,
        chimeras: List[ChimeraSample],
        onehot: bool = True,
        max_len: int = None,
        pad_to_max: bool = True,
    ):
        """
        Args:
            chimeras: 伪嵌合体列表
            onehot: 是否生成一键编码
            max_len: 最大序列长度（超过则截断，不足则补零）
            pad_to_max: 是否 padding 到 max_len
        """
        self.chimeras = chimeras
        self.onehot = onehot
        self.max_len = max_len
        self.pad_to_max = pad_to_max
        
        if max_len is None and pad_to_max:
            self.max_len = max(c.lengths['total'] for c in chimeras)
    
    def __len__(self):
        return len(self.chimeras)
    
    def __getitem__(self, idx: int):
        chimera = self.chimeras[idx]
        L = chimera.lengths['total']
        target_len = self.max_len if self.pad_to_max else L
        
        sequence = chimera.sequence
        contact_map = chimera.contact_map
        domain_labels = chimera.domain_labels
        
        # Padding / Truncation
        if self.pad_to_max:
            sequence = sequence[:target_len].ljust(target_len, 'N')
            
            padded_cmap = np.zeros((target_len, target_len), dtype=np.float32)
            padded_cmap[:L, :L] = contact_map[:target_len, :target_len]
            contact_map = padded_cmap
            
            padded_domain = np.zeros(target_len, dtype=np.int32)
            padded_domain[:L] = domain_labels[:target_len]
            domain_labels = padded_domain
        
        # 有效位置 mask
        valid_mask = np.ones((target_len, target_len), dtype=np.float32)
        if self.pad_to_max and L < target_len:
            valid_mask[L:, :] = 0
            valid_mask[:, L:] = 0
        
        # 一键编码
        onehot_seq = None
        if self.onehot:
            onehot_seq = np.zeros((target_len, 4), dtype=np.float32)
            for i, base in enumerate(sequence[:target_len]):
                if base in self.BASE_TO_IDX:
                    onehot_seq[i, self.BASE_TO_IDX[base]] = 1.0
        
        return {
            'sequence': sequence[:target_len],
            'contact_map': contact_map,
            'domain_labels': domain_labels,
            'onehot': onehot_seq,
            'valid_mask': valid_mask,
            'length': min(L, target_len),
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, np.ndarray]:
        """自定义 collate 函数，将所有字段堆叠为 batch"""
        import torch
        
        collated = {}
        for key in batch[0].keys():
            if key == 'sequence':
                collated[key] = [item[key] for item in batch]
            elif key == 'length':
                collated[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
            elif batch[0][key] is not None:
                collated[key] = torch.tensor(
                    np.stack([item[key] for item in batch]), dtype=torch.float32
                )
            else:
                collated[key] = None
        return collated


# ============================================================================
# 统计与可视化
# ============================================================================

def compute_statistics(chimeras: List[ChimeraSample]) -> Dict:
    """计算数据集的详细统计信息"""
    total_lens = [c.lengths['total'] for c in chimeras]
    trna5_lens = [c.lengths['trna_5'] for c in chimeras]
    insert_lens = [c.lengths['insert'] for c in chimeras]
    trna3_lens = [c.lengths['trna_3'] for c in chimeras]
    num_pairs = []
    cross_pairs = []
    trna5_pairs = []
    insert_pairs = []
    trna3_pairs = []
    
    for c in chimeras:
        L5 = c.lengths['trna_5']
        L3 = c.lengths['trna_3']
        ins = c.lengths['insert']
        total = c.lengths['total']
        
        cmap = c.contact_map
        
        num_pairs.append(cmap.sum() / 2)
        cross_pairs.append(cmap[:L5, L5+ins:].sum())
        trna5_pairs.append(cmap[:L5, :L5].sum() / 2)
        insert_pairs.append(cmap[L5:L5+ins, L5:L5+ins].sum() / 2)
        trna3_pairs.append(cmap[L5+ins:, L5+ins:].sum() / 2)
    
    # GC 含量
    gc_contents = []
    for c in chimeras:
        seq = c.sequence
        gc = (seq.count('G') + seq.count('C')) / len(seq) if len(seq) > 0 else 0
        gc_contents.append(gc)
    
    # 长程配对距离分布
    long_range_distances = []
    for c in chimeras:
        cmap = c.contact_map
        L5 = c.lengths['trna_5']
        ins = c.lengths['insert']
        # 只统计 5'↔3' 跨域配对
        cross_region = cmap[:L5, L5+ins:]
        rows, cols = np.where(cross_region > 0.5)
        for r, c_idx in zip(rows, cols):
            # 在完整序列中的距离
            distance = (L5 + ins + c_idx) - r
            long_range_distances.append(int(distance))
    
    stats = {
        'num_samples': len(chimeras),
        'total_length': {
            'mean': np.mean(total_lens), 'std': np.std(total_lens),
            'min': min(total_lens), 'max': max(total_lens),
        },
        'trna_5_length': {
            'mean': np.mean(trna5_lens), 'std': np.std(trna5_lens),
        },
        'insert_length': {
            'mean': np.mean(insert_lens), 'std': np.std(insert_lens),
        },
        'trna_3_length': {
            'mean': np.mean(trna3_lens), 'std': np.std(trna3_lens),
        },
        'pairs': {
            'total_mean': np.mean(num_pairs),
            'total_std': np.std(num_pairs),
            'cross_mean': np.mean(cross_pairs),
            'cross_std': np.std(cross_pairs),
            'trna_5_mean': np.mean(trna5_pairs),
            'insert_mean': np.mean(insert_pairs),
            'trna_3_mean': np.mean(trna3_pairs),
        },
        'gc_content': {
            'mean': np.mean(gc_contents),
            'std': np.std(gc_contents),
        },
        'cross_pair_ratio': sum(1 for p in cross_pairs if p > 0) / len(cross_pairs),
        'long_range_distances': {
            'mean': np.mean(long_range_distances) if long_range_distances else 0,
            'median': np.median(long_range_distances) if long_range_distances else 0,
            'min': min(long_range_distances) if long_range_distances else 0,
            'max': max(long_range_distances) if long_range_distances else 0,
        },
    }
    return stats


def print_statistics(chimeras: List[ChimeraSample]):
    """打印统计信息"""
    stats = compute_statistics(chimeras)
    
    print("\n" + "=" * 60)
    print("伪嵌合体数据集统计")
    print("=" * 60)
    print(f"  总样本数:       {stats['num_samples']}")
    print(f"  总长度:         {stats['total_length']['mean']:.1f} ± "
          f"{stats['total_length']['std']:.1f} "
          f"[{stats['total_length']['min']}, {stats['total_length']['max']}]")
    print(f"  5' tRNA 长度:   {stats['trna_5_length']['mean']:.1f} ± "
          f"{stats['trna_5_length']['std']:.1f}")
    print(f"  Insert 长度:    {stats['insert_length']['mean']:.1f} ± "
          f"{stats['insert_length']['std']:.1f}")
    print(f"  3' tRNA 长度:   {stats['trna_3_length']['mean']:.1f} ± "
          f"{stats['trna_3_length']['std']:.1f}")
    print(f"  平均配对对数:   {stats['pairs']['total_mean']:.1f} ± "
          f"{stats['pairs']['total_std']:.1f}")
    print(f"  - 5' tRNA 内:   {stats['pairs']['trna_5_mean']:.1f}")
    print(f"  - Insert 内:    {stats['pairs']['insert_mean']:.1f}")
    print(f"  - 3' tRNA 内:   {stats['pairs']['trna_3_mean']:.1f}")
    print(f"  - 跨域长程:     {stats['pairs']['cross_mean']:.1f} ± "
          f"{stats['pairs']['cross_std']:.1f}")
    print(f"  GC 含量:        {stats['gc_content']['mean']:.3f} ± "
          f"{stats['gc_content']['std']:.3f}")
    print(f"  有长程配对比例: {stats['cross_pair_ratio']:.2%}")
    if stats['long_range_distances']['mean'] > 0:
        print(f"  长程配对距离:   mean={stats['long_range_distances']['mean']:.0f}, "
              f"median={stats['long_range_distances']['median']:.0f}")
    print("=" * 60)


def visualize_sample(chimera: ChimeraSample, output_path: str = None):
    """
    可视化一条伪嵌合体样本的 contact map。
    需要 matplotlib，如果不可用则跳过。
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[visualize] matplotlib 未安装，跳过可视化")
        return
    
    cmap = chimera.contact_map
    L5 = chimera.lengths['trna_5']
    ins = chimera.lengths['insert']
    L3 = chimera.lengths['trna_3']
    total = chimera.lengths['total']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图: Contact Map
    ax = axes[0]
    ax.imshow(cmap, cmap='Blues', aspect='auto', interpolation='none')
    
    # 域分隔线
    ax.axhline(y=L5 - 0.5, color='red', linewidth=1.5, linestyle='--')
    ax.axhline(y=L5 + ins - 0.5, color='red', linewidth=1.5, linestyle='--')
    ax.axvline(x=L5 - 0.5, color='red', linewidth=1.5, linestyle='--')
    ax.axvline(x=L5 + ins - 0.5, color='red', linewidth=1.5, linestyle='--')
    
    # 标注区域
    ax.text(L5/2, -3, "5' tRNA", ha='center', fontsize=9, color='red')
    ax.text(L5 + ins/2, -3, "Insert", ha='center', fontsize=9, color='red')
    ax.text(L5 + ins + L3/2, -3, "3' tRNA", ha='center', fontsize=9, color='red')
    
    ax.set_title(f"Contact Map ({total}nt, {int(cmap.sum()/2)} pairs)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Position")
    
    # 右图: 统计信息
    ax = axes[1]
    ax.axis('off')
    
    info_lines = [
        f"tRNA: {chimera.trna_name}",
        f"Insert: {chimera.insert_name}",
        f"",
        f"Length: {total} nt",
        f"  5' tRNA: {L5} nt",
        f"  Insert:  {ins} nt",
        f"  3' tRNA: {L3} nt",
        f"",
        f"Total pairs: {int(cmap.sum()/2)}",
        f"  5' internal:    {int(cmap[:L5,:L5].sum()/2)}",
        f"  Insert internal: {int(cmap[L5:L5+ins,L5:L5+ins].sum()/2)}",
        f"  3' internal:    {int(cmap[L5+ins:,L5+ins:].sum()/2)}",
        f"  5'↔3' cross:    {int(cmap[:L5,L5+ins:].sum())}",
        f"",
        f"Sequence (first 60nt):",
        f"  {chimera.sequence[:60]}",
        f"  ...",
    ]
    
    for i, line in enumerate(info_lines):
        ax.text(0.05, 0.95 - i * 0.05, line, transform=ax.transAxes,
                fontsize=10, family='monospace', verticalalignment='top')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[visualize] 保存到 {output_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# 保存与加载
# ============================================================================

def save_dataset(chimeras: List[ChimeraSample], output_path: str):
    """保存数据集为 pickle"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
                exist_ok=True)
    
    # 转为可序列化格式
    save_data = []
    for c in chimeras:
        save_data.append({
            'sequence': c.sequence,
            'contact_map': c.contact_map,
            'domain_labels': c.domain_labels,
            'trna_seq_5': c.trna_seq_5,
            'trna_seq_3': c.trna_seq_3,
            'insert_seq': c.insert_seq,
            'trna_name': c.trna_name,
            'insert_name': c.insert_name,
            'lengths': c.lengths,
        })
    
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[save] {len(save_data)} 条 → {output_path} ({file_size:.1f} MB)")


def load_dataset(filepath: str) -> List[ChimeraSample]:
    """从 pickle 加载数据集"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    chimeras = []
    for d in data:
        chimeras.append(ChimeraSample(
            sequence=d['sequence'],
            contact_map=d['contact_map'],
            domain_labels=d['domain_labels'],
            trna_seq_5=d['trna_seq_5'],
            trna_seq_3=d['trna_seq_3'],
            insert_seq=d['insert_seq'],
            trna_name=d['trna_name'],
            insert_name=d['insert_name'],
            lengths=d['lengths'],
        ))
    print(f"[load] {len(chimeras)} 条 → {filepath}")
    return chimeras


def split_dataset(chimeras: List[ChimeraSample],
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1,
                  seed: int = 42) -> Tuple[List[ChimeraSample], ...]:
    """
    将数据集划分为 train/val/test。
    
    使用 tRNA 名称做分层划分，确保同一 tRNA 来源的样本在同一子集中。
    """
    random.seed(seed)
    
    # 按 tRNA 名称分组
    by_trna = defaultdict(list)
    for c in chimeras:
        by_trna[c.trna_name].append(c)
    
    trna_names = list(by_trna.keys())
    random.shuffle(trna_names)
    
    n = len(trna_names)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_names = set(trna_names[:train_end])
    val_names = set(trna_names[train_end:val_end])
    test_names = set(trna_names[val_end:])
    
    train = [c for c in chimeras if c.trna_name in train_names]
    val = [c for c in chimeras if c.trna_name in val_names]
    test = [c for c in chimeras if c.trna_name in test_names]
    
    print(f"[split] train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='伪 tRNA-miRNA 嵌合体数据生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法
  python pseudo_chimera_generator.py --bprna_file bpRNA.txt --output chimeras.pkl
  
  # 指定参数
  python pseudo_chimera_generator.py \\
      --bprna_file bpRNA.txt \\
      --output chimeras_train.pkl \\
      --num_samples 8000 \\
      --trna_5prime_len 43 --trna_3prime_len 15 \\
      --insert_min 100 --insert_max 150 \\
      --seed 42
  
  # 使用 ViennaRNA 预测插入片段结构
  python pseudo_chimera_generator.py \\
      --bprna_file bpRNA.txt \\
      --output chimeras.pkl \\
      --use_vienna --vienna_path RNAfold
  
  # 同时保存 train/val/test 三个文件
  python pseudo_chimera_generator.py \\
      --bprna_file bpRNA.txt \\
      --output chimeras.pkl \\
      --split_output --split_ratios 0.8,0.1,0.1
        """
    )
    
    # 必选参数
    parser.add_argument('--bprna_file', type=str, required=True,
                        help='bpRNA 格式数据文件路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出文件路径 (.pkl)')
    
    # 可选参数
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='生成样本数 (默认: 5000)')
    parser.add_argument('--trna_5prime_len', type=int, default=43,
                        help="5' tRNA 片段长度 (默认: 43)")
    parser.add_argument('--trna_3prime_len', type=int, default=15,
                        help="3' tRNA 片段长度 (默认: 15)")
    parser.add_argument('--insert_min', type=int, default=100,
                        help='插入片段最小长度 (默认: 100)')
    parser.add_argument('--insert_max', type=int, default=150,
                        help='插入片段最大长度 (默认: 150)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    # tRNA 识别
    parser.add_argument('--tRNA_keyword', type=str, default='tRNA',
                        help='tRNA 识别关键词，多词用逗号分隔 (默认: tRNA)')
    
    # ViennaRNA
    parser.add_argument('--use_vienna', action='store_true',
                        help='使用 ViennaRNA 预测无结构数据的 insert 片段')
    parser.add_argument('--vienna_path', type=str, default='RNAfold',
                        help='RNAfold 可执行文件路径 (默认: RNAfold)')
    
    # 输出选项
    parser.add_argument('--split_output', action='store_true',
                        help='同时输出 train/val/test 三个文件')
    parser.add_argument('--split_ratios', type=str, default='0.8,0.1,0.1',
                        help='train/val/test 比例 (默认: 0.8,0.1,0.1)')
    parser.add_argument('--visualize', type=int, default=0,
                        help='可视化 N 个样本 (默认: 0, 不可视化)')
    
    # 加载限制
    parser.add_argument('--max_load', type=int, default=None,
                        help='最大加载条目数 (用于测试)')
    
    args = parser.parse_args()
    
    # ========================================
    # 1. 加载数据
    # ========================================
    print("=" * 60)
    print("伪嵌合体数据生成器")
    print("=" * 60)
    print(f"数据文件: {args.bprna_file}")
    
    entries = parse_bprna_file(args.bprna_file, max_entries=args.max_load)
    
    if len(entries) == 0:
        print("[错误] 未加载到任何数据，请检查文件路径和格式")
        sys.exit(1)
    
    # ========================================
    # 2. 分类 tRNA / 非tRNA
    # ========================================
    trna_entries, non_trna_entries = classify_entries(entries, args.tRNA_keyword)
    
    if len(trna_entries) < 5:
        print(f"[错误] tRNA 数量过少 ({len(trna_entries)}), 需要 >= 5")
        print(f"  提示: 尝试使用 --tRNA_keyword 指定更宽泛的关键词")
        sys.exit(1)
    
    if len(non_trna_entries) < 10:
        print(f"[错误] 非 tRNA 数量过少 ({len(non_trna_entries)}), 需要 >= 10")
        sys.exit(1)
    
    # ========================================
    # 3. 生成伪嵌合体
    # ========================================
    generator = PseudoChimeraGenerator(
        trna_entries=trna_entries,
        insert_entries=non_trna_entries,
        trna_5prime_len=args.trna_5prime_len,
        trna_3prime_len=args.trna_3prime_len,
        insert_len_min=args.insert_min,
        insert_len_max=args.insert_max,
        seed=args.seed,
        use_vienna=args.use_vienna,
        vienna_path=args.vienna_path,
    )
    
    chimeras = generator.generate(args.num_samples)
    
    if len(chimeras) == 0:
        print("[错误] 未生成任何有效嵌合体")
        sys.exit(1)
    
    # ========================================
    # 4. 统计
    # ========================================
    print_statistics(chimeras)
    
    # ========================================
    # 5. 保存
    # ========================================
    if args.split_output:
        ratios = [float(r) for r in args.split_ratios.split(',')]
        if len(ratios) != 3:
            ratios = [0.8, 0.1, 0.1]
        
        train, val, test = split_dataset(
            chimeras, ratios[0], ratios[1], ratios[2], seed=args.seed
        )
        
        base, ext = os.path.splitext(args.output)
        save_dataset(train, f"{base}_train{ext}")
        save_dataset(val, f"{base}_val{ext}")
        save_dataset(test, f"{base}_test{ext}")
    else:
        save_dataset(chimeras, args.output)
    
    # ========================================
    # 6. 可视化
    # ========================================
    if args.visualize > 0:
        n_viz = min(args.visualize, len(chimeras))
        for i in random.sample(range(len(chimeras)), n_viz):
            viz_path = os.path.splitext(args.output)[0] + f"_sample{i:03d}.png"
            visualize_sample(chimeras[i], viz_path)
    
    print("\n完成!")


if __name__ == '__main__':
    main()

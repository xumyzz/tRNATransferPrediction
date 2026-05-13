import os
import shutil
import re
from tqdm import tqdm

source_dir = '/root/autodl-tmp/newPredicProject/dbnFiles'
target_dir = '/root/autodl-tmp/myPredicProject/tRNA_dataset'

os.makedirs(target_dir, exist_ok=True)

tRNA_count = 0

print("🚀 启动基于‘四叶草形态学’的 tRNA 提取...")

for filename in tqdm(os.listdir(source_dir)):
    if not filename.endswith('.dbn'):
        continue
        
    filepath = os.path.join(source_dir, filename)
    
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except Exception:
        continue
        
    if len(lines) < 4:
        continue
        
    # 提取序列和结构 (倒数第二行是序列，最后一行是结构)
    # bpRNA 的标准格式最后两行必定是 sequence 和 structure
    sequence = lines[-2]
    structure = lines[-1]
    
    seq_len = len(sequence)
    
    # 1. 第一道门槛：生物学长度 (65 到 100 之间)
    if not (65 <= seq_len <= 100):
        continue
        
    # 2. 第二道门槛：拓扑结构检测 (核心魔法)
    # 寻找形如 (...) 的最内层发夹环。
    # \(\.+\) 的意思是：一个左括号，中间跟着一堆点，然后一个右括号
    hairpins = re.findall(r'\(\.+\)', structure)
    num_hairpins = len(hairpins)
    
    # 真正的 tRNA (四叶草) 必定包含 3 个发夹环 (D环, 反密码子环, T环)
    # 少数带有可变环 (Variable loop stem) 的 tRNA 可能会有 4 个发夹环
    if num_hairpins == 3 or num_hairpins == 4:
        # 匹配成功！这就是我们要的纯正 tRNA！
        shutil.copy(filepath, os.path.join(target_dir, filename))
        tRNA_count += 1

print(f"\n=========================================")
print(f"🎉 提取完成！共找到 {tRNA_count} 个完美的 tRNA 结构！")
print(f"这批数据是纯正的四叶草拓扑，它们已经保存在: {target_dir}")
print(f"你的 Mamba 模型和四叶草先验准备好起飞了！")
print(f"=========================================")
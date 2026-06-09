import os
import subprocess
import shutil
from tqdm import tqdm

# === 路径配置区 (请核对你的实际路径) ===
# 你的训练集目录 (提取出的 bpRNA tRNA)
train_dir = '/root/autodl-tmp/myPredicProject/tRNA_dataset'
# 你的原始测试集目录 (刚才转换的 ArchiveII dbn)
test_dir = '/root/autodl-tmp/myPredicProject/ArchiveIIData/ArchiveII_testset'
# 洗白后的绝对零样本测试集 (我们将生成这个新目录)
strict_test_dir = '/root/autodl-tmp/myPredicProject/ArchiveII_Strict_ZeroShot'

# 临时工作空间
os.makedirs('temp_filter', exist_ok=True)
train_fasta = 'temp_filter/train.fasta'
test_fasta = 'temp_filter/test.fasta'
filtered_fasta = 'temp_filter/filtered_test'

def dbn_to_fasta(src_dir, out_fasta):
    with open(out_fasta, 'w') as out_f:
        for filename in os.listdir(src_dir):
            if filename.endswith('.dbn'):
                with open(os.path.join(src_dir, filename), 'r') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                    if len(lines) >= 4:
                        seq = lines[-2]
                        # 把 T 转成 U，转大写，防止因为大小写导致的漏网之鱼
                        seq = seq.upper().replace('T', 'U') 
                        out_f.write(f">{filename}\n{seq}\n")

print("1. 正在将训练集和测试集打包成 FASTA (以便送入 CD-HIT 审判)...")
dbn_to_fasta(train_dir, train_fasta)
dbn_to_fasta(test_dir, test_fasta)

print("\n2. 启动 cd-hit-est-2d 进行跨域查重 (阈值 0.8)...")
# cd-hit-est-2d 会把 db2(测试集) 中与 db1(训练集) 相似度 >= 80% 的序列统统删掉！
cmd = f"cd-hit-est-2d -i {train_fasta} -i2 {test_fasta} -o {filtered_fasta} -c 0.8 -n 5 -d 0"
subprocess.run(cmd, shell=True, check=True)

print("\n3. 提取幸存者 (真正的 Zero-Shot 数据)...")
os.makedirs(strict_test_dir, exist_ok=True)
survivors = 0

# 读取 cd-hit 留下来的纯洁序列名单
with open(filtered_fasta, 'r') as f:
    for line in f:
        if line.startswith('>'):
            dbn_filename = line.strip()[1:] # 去掉 >
            src_file = os.path.join(test_dir, dbn_filename)
            dst_file = os.path.join(strict_test_dir, dbn_filename)
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)
                survivors += 1

print("\n" + "="*50)
print(f"⚖️  残酷审判结束！")
print(f"ArchiveII 原有数据: {len(os.listdir(test_dir))} 条")
print(f"被判定为数据泄露并删除: {len(os.listdir(test_dir)) - survivors} 条")
print(f"🎉 幸存的绝对纯洁数据: {survivors} 条")
print(f"这 {survivors} 条数据已经保存在: {strict_test_dir}")
print("="*50)
print("现在，用你的 model_best.pth 去测试这个新文件夹！哪怕 F1 掉到 0.85，这也是真正的 SOTA！")
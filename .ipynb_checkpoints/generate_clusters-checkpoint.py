import os
import subprocess

# === 路径配置区 (请核对你的实际路径) ===
# 刚刚洗白后的大库文件夹路径
clean_dbn_dir = '/root/autodl-tmp/myPredicProject/FinalTestData/PretrainData'

# 我们准备生成的工作目���
output_dir = '/root/autodl-tmp/myPredicProject/FinalTestData/PretrainDataclusterfile'

# 确保文件夹存在 (你之前报错就是因为不小心把这行和上一行揉在一起了)
os.makedirs(output_dir, exist_ok=True)

# 中间产物和最终产物路径
fasta_path = os.path.join(output_dir, 'all_clean_rna.fasta')
cluster_out = os.path.join(output_dir, 'rna_clusters')
# cd-hit 会自动生成 rna_clusters.clstr 文件

def dbns_to_fasta(src_dir, out_fasta):
    print(f"1. 正在读取 {src_dir} 的所有文件并打包为 FASTA...")
    count = 0
    with open(out_fasta, 'w') as out_f:
        for filename in os.listdir(src_dir):
            if filename.endswith('.dbn'):
                filepath = os.path.join(src_dir, filename)
                with open(filepath, 'r') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                    if len(lines) >= 4:
                        # 提取序列并统一格式
                        seq = lines[-2].upper().replace('T', 'U') 
                        # 这里的 header 必须和 dbn 文件名一致，后续代码才能按名字对上号
                        out_f.write(f">{filename}\n{seq}\n")
                        count += 1
    print(f"✅ 成功打包 {count} 条序列！")
    return count

# 执行打包
dbns_to_fasta(clean_dbn_dir, fasta_path)

# 启动 CD-HIT-EST 进行内部聚类
print("\n2. 启动 CD-HIT-EST 进行内部聚类 (阈值 80%)...")
print("   (这可能需要几分钟时间，请耐心等待...)")

# -c 0.8: 80% 相似度归为一类
# -n 5: 配合 0.8 阈值的字长
# -M 0: 不限制内存使用，加速运算
# -d 0: 完整保留序列名称
cmd = f"cd-hit-est -i {fasta_path} -o {cluster_out} -c 0.8 -n 5 -M 0 -d 0"
subprocess.run(cmd, shell=True, check=True)

print("\n" + "="*50)
print(f"🎉 聚类完成！")
print(f"你的聚类文件 (用于训练切分) 已经生成在: {cluster_out}.clstr")
print("="*50)用于训练切分) 已经生成在: {cluster_out}.clstr")
print("="*50)
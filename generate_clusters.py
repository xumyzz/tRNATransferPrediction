import os
import subprocess

# === 路径配置区 ===
clean_dbn_dir = '/root/autodl-tmp/myPredicProject/FinalTestData/PretrainData'
output_dir = '/root/autodl-tmp/myPredicProject/FinalTestData/PretrainDataclusterfile'

os.makedirs(output_dir, exist_ok=True)

fasta_path = os.path.join(output_dir, 'all_clean_rna.fasta')
cluster_out = os.path.join(output_dir, 'rna_clusters')

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
                        seq = lines[-2].upper().replace('T', 'U') 
                        out_f.write(f">{filename}\n{seq}\n")
                        count += 1
    print(f"✅ 成功打包 {count} 条序列！")
    return count

dbns_to_fasta(clean_dbn_dir, fasta_path)

print("\n2. 启动 CD-HIT-EST 进行内部聚类 (阈值 80%)...")
cmd = f"cd-hit-est -i {fasta_path} -o {cluster_out} -c 0.8 -n 5 -M 0 -d 0"
subprocess.run(cmd, shell=True, check=True)

print("\n" + "="*50)
print(f"🎉 聚类完成！")
print(f"你的聚类文件已经生成在: {cluster_out}.clstr")
print("="*50)

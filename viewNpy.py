import numpy as np

# 加载一条结果
cmap = np.load('chimera_predictions/bpRNA_CRW_16422.npy')
print(f"形状: {cmap.shape}")
print(f"配对总数: {int(cmap.sum() / 2)}")

# 看具体哪些位置配对了
pairs = list(zip(*np.where(np.triu(cmap) > 0)))
print(f"配对位置: {pairs[:20]}")  # 前 20 对
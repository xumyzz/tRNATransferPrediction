import pandas as pd
import RNA

df_trna = pd.read_csv("/root/autodl-tmp/myPredicProject/tRNA_mRNA_Binding_data/tRNA_list.csv")
df_mirna = pd.read_csv("/root/autodl-tmp/myPredicProject/tRNA_mRNA_Binding_data/miRNA_list.csv")
linker_seq = "AAAAA"

results = []
for _, row_t in df_trna.iterrows():
    for _, row_m in df_mirna.iterrows():
        chimeric_seq = row_t['Sequence'] + linker_seq + row_m['Sequence']
        
        # 纯粹的、没有任何 DL 干扰的物理折叠
        struct, energy = RNA.fold(chimeric_seq)
        
        results.append({
            "Chimera_Name": f"{row_t['Name']}_{row_m['Name']}",
            "Predicted_Structure": struct,
            "Real_Energy (kcal/mol)": round(energy, 2)
        })

df_results = pd.DataFrame(results)
df_results.to_excel("/root/autodl-tmp/myPredicProject/tRNA_mRNA_Binding_data/Pure_Physical_Baseline.xlsx", index=False)
print("纯物理预测跑完了，快去看看传统算法错得有多离谱！")
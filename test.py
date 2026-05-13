import RNA

my_chimeric_seq = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUUUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCAUGUGGUCGACAGGUGUAUGAAGACUGUCACGGGCAAGUUGCGGAA"

# 没有任何 DL 约束，纯粹靠物理热力学预测
struct_pure_physics, energy_pure = RNA.fold(my_chimeric_seq)

print("纯物理预测结构:", struct_pure_physics)
print("你融合DL的结构:", "(((((((..((((........))))((((((.......))))))....(((((.......)))))))))))).((.(((((((..((......))..))))).)).))(((...)))....")
import pandas as pd
from constant.constants import FILE_CANDIDATE_POOL_ADDRESS

# 读取 CSV
df = pd.read_csv(FILE_CANDIDATE_POOL_ADDRESS)

# 根据 correct_passage_index 获取正确答案内容
df["correct_passage"] = df.apply(
    lambda row: row[f"passage_{row['correct_passage_index']}"], axis=1
)

# 只保留需要的列
result_df = df[["id", "query", "correct_passage"]]

# 保存到新的 CSV
result_df.to_csv("Ground-truth/QACorrect_Passages.csv", index=False)

print("提取完成，已保存到 QACorrect_Passages.csv")

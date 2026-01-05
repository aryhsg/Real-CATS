import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

"""

這個檔案在做的事情：
1. 讀取經過初步處理的比特幣資料集 (bitcoin_data_features.tsv)
2. 定義並刪除冗餘欄位，以減少特徵數量
3. 對數據進行標準化處理，使其均值為0，標準差為1
4. 計算欄位之間的相關性矩陣，並輸出至CSV檔案
5. 繪製相關性熱圖
6. 儲存最終處理後的數據到新的TSV檔案(bitcoin_data_final_features.tsv)

"""

features = pd.read_csv("bitcoin_data_features2.tsv", sep="\t")

# 定義要刪除的冗餘欄位
drop_redundant_cols = [
    # USD 類
    'total_received_USD', 'total_sent_USD', 'received_Variance_USD', 'sent_Variance_USD',
    # 重複的次數類
    'payment_transactions', 'receipt_transactions', 'total_output_slots', 'total_input_slots',
    # 重複的活躍度
    'activity_w', 'activity_d',
    # 與總額高度相關的極值
    'max_sent_amount', 'min_sent_amount', 'max_received_amount', 'min_received_amount'
]

columns_final = features.drop(columns=drop_redundant_cols)

print(f"最終特徵數: {len(columns_final.columns)}")
print("模型將使用的最終特徵清單：", columns_final.columns.tolist())


# 標準化參數
scaler = StandardScaler()
# 對數轉換後進行標準化
columns_final.iloc[:, :] = scaler.fit_transform(columns_final.iloc[:, :])

print("標準化後的特徵範例：")
print(columns_final.head())


# 計算欄位之間的相關性矩陣，並輸出至CSV檔案
corrlatiomn_matrix = columns_final.corr()
corrlatiomn_matrix.to_csv("correlation_matrix_after_standardization.csv")
# 繪製相關性熱圖
plt.figure(figsize=(15, 12))
sns.heatmap(corrlatiomn_matrix,annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Correlation Heatmap after Standardization', fontsize=16)
plt.xticks(rotation=30, ha='right')
plt.show()


# 儲存最終處理後的數據
columns_final.to_csv("x_train.tsv", index=False, sep="\t")
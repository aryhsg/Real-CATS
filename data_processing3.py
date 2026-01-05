import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

"""

這個檔案在做的事情：
1. 讀取經過初步處理的比特幣資料集 (eth_data_features.tsv)
2. 定義並刪除冗餘欄位，以減少特徵數量
3. 對數據進行標準化處理，使其均值為0，標準差為1
4. 計算欄位之間的相關性矩陣，並輸出至CSV檔案
5. 繪製相關性熱圖
6. 儲存最終處理後的數據到新的TSV檔案(eth_data_final_features.tsv)

"""

features = pd.read_csv("eth_data_features2.tsv", sep="\t")

# 定義要刪除的冗餘欄位
drop_redundant_cols = [
    # 1. 幣別重疊 (ETH 已經足以代表交易規模，USD 只是匯率轉換的衍生值)
    'total_received_USD', 
    'total_sent_USD', 
    'received_Variance_USD', 
    'sent_Variance_USD',

    # 2. 統計加總重疊 (transaction_number 等於收與發的總和，拆開看更有辨識度)
    'transaction_number', 
    
    # 3. Gas 相關冗餘 (Used 與 Limit 高度正相關，通常保留實際消耗量 Used)
    'total_gas_limit',

    # 4. 時間數據冗餘 (activity_time 通常與 lifetime 高度線性相關)
    'activity_time',

    # 5. 外部標記 (標籤洩漏風險：這些是事後審核結果，放入訓練會讓模型「作弊」)
    'etherscan_checked',

    "from_contract_wd_internal", 
    "from_EOA_wd_internal",
    "to_contract_wd_internal",
    "to_contract_wd_normal",
    "to_contract_wod_normal",
    "to_EOA_wd_internal", 
    "to_EOA_wod_internal" 

]


final_drop = [
    'transaction_fee', 'total_gasP_set', # 與 total_gas_used 重疊
    'max_received_amount', 'max_sent_amount', # 與總量重疊
    'transaction_fee_Variance', 'gas_limit_Variance' # 與 gasP_set_Variance 重疊
]


columns_final = features.drop(columns=drop_redundant_cols)
columns_final2 = columns_final.drop(columns=final_drop)

print(f"最終特徵數: {len(columns_final2.columns)}")
print("模型將使用的最終特徵清單：", columns_final2.columns.tolist())


# 標準化參數
scaler = StandardScaler()
# 對數轉換後進行標準化
columns_final2.iloc[:, :] = scaler.fit_transform(columns_final2.iloc[:, :])

print("標準化後的特徵範例：")
print(columns_final2.head())
# 計算欄位之間的相關性矩陣，並輸出至CSV檔案
corrlatiomn_matrix = columns_final2.corr()
corrlatiomn_matrix.to_csv("correlation_matrix_after_standardization.csv")
# 繪製相關性熱圖
plt.figure(figsize=(20, 16))
sns.heatmap(corrlatiomn_matrix,annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Correlation Heatmap after Standardization', fontsize=16)
plt.xticks(rotation=30, ha='right')
plt.show()


# 儲存最終處理後的數據
columns_final2.to_csv("x_train2.tsv", index=False, sep="\t")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

"""

這個檔案在做的事情：
1. 讀取合併後的比特幣資料集 (bitcoin_data.tsv)
2. 淘汰不須放入模型之欄位
3. 檢查並填補 NaN 值
4. 針對分布較極端的欄位，進行對數轉換 以減少偏態
5. 計算欄位之間的相關性矩陣，並輸出至CSV檔案
6. 繪製相關性熱圖
7. 儲存處理後的數據到新的TSV檔案(bitcoin_data_features.tsv)

"""

bitcoin_data = pd.read_csv("bitcoin_data_processed.tsv", sep="\t")


# 提取標籤欄位，並儲存為y_train.tsv
y_train = bitcoin_data["label_idx"].values
y_train = pd.DataFrame(y_train, columns=['label_idx'])
y_stat = y_train['label_idx'].value_counts().sort_index()
y_stat.to_csv("y_train_stats.csv")
y_train.to_csv("y_train.tsv", index=False, sep="\t")


#print(bitcoin_data.columns.tolist())
# 淘汰不須放入模型之欄位
drop_cols = [
    'address', 'label', 'criminal', 'label_idx',           # 標籤與識別碼
    'first_time', 'last_time',                             # 原始時間戳
    'max_sent_transaction_id', 'min_sent_transaction_id',  # 交易ID
    'max_received_transaction_id', 'min_received_transaction_id',
    'watchback_checked', 'gs_checked'                      # 內部註記
]


columns_to_use = bitcoin_data.drop(columns=drop_cols)
"""print(f"原始欄位數: {len(bitcoin_data.columns)}")
print(f"篩選後特徵數: {len(columns_to_use.columns)}")
print("模型將使用的特徵清單：", columns_to_use.columns.tolist())"""

def check_nan(array):
    nan_count = np.sum(pd.isna(array))
    print(f"NaN count: {nan_count}")
    if nan_count > 0:
        print("Indices of NaN values:", np.where(pd.isna(array))[0])

# 檢查每個欄位的 NaN 狀況
for value in columns_to_use.columns:
    print(f"Checking NaN for {value}:")
    check_nan(columns_to_use[value])
    print("\n")


# 輸出特定欄位的基本統計資訊
columns_stats_data_vertical = pd.Series(dtype='float64')
for column in columns_to_use.columns:
    columns_stats_data = columns_to_use[column].describe()
    columns_stats_data_vertical = pd.concat((columns_stats_data_vertical, columns_stats_data), axis=1)
columns_stats_dataframe = columns_stats_data_vertical.T
columns_stats_dataframe = columns_stats_dataframe.drop(0)
# 儲存統計資訊至CSV檔案
columns_stats_dataframe.to_csv("column_stats.csv")


# 將 NaN 值填補為 0
def fill_nan_with_zero(array):
    return array.fillna(0)

for value in columns_to_use.columns:
    columns_to_use[value] = fill_nan_with_zero(columns_to_use[value])
    print(f"After filling NaN with zero for {value}:")
    check_nan(columns_to_use[value])
    print("\n")


# 對數轉換函數
def Log_transform(x):
    if x > 0:
        return np.log1p(x)
    else:
        return 0
    
# 針對分布較極端的欄位，進行對數轉換 以減少偏態
columns_stats_data_vertical = pd.Series(dtype='float64')
for column in columns_to_use.columns:
    columns_to_use[column] = columns_to_use[column].apply(Log_transform)
# 輸出特定欄位的基本統計資訊
    columns_stats_data = columns_to_use[column].describe()
    columns_stats_data_vertical = pd.concat((columns_stats_data_vertical, columns_stats_data), axis=1)
columns_stats_dataframe = columns_stats_data_vertical.T
log_columns_stats_dataframe = columns_stats_dataframe.drop(0)
# 儲存統計資訊至CSV檔案
log_columns_stats_dataframe.to_csv("log_column_stats.csv")


# 計算欄位之間的相關性矩陣，並輸出至CSV檔案
corrlatiomn_matrix = columns_to_use.corr()
corrlatiomn_matrix.to_csv("correlation_matrix.csv")


# 繪製相關性熱圖
plt.figure(figsize=(15, 12))
sns.heatmap(corrlatiomn_matrix,annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Correlation Heatmap after Log Transformation', fontsize=16)
plt.xticks(rotation=30, ha='right')
plt.show()


# 繪製各個類別的餘額分佈 (以transaction_fee為例)
# kde=True 會在柱狀圖上加上那條平滑的「鐘形」曲線
"""
plt.figure(figsize=(15, 8))
sns.histplot(columns_to_use['transaction_fee'], kde=False, bins=50, color='skyblue')
plt.title('Transaction Fee Distribution')
plt.xlabel('transaction_fee')
plt.ylabel('dollar ')
plt.show()"""


# 儲存處理後的數據
columns_to_use.to_csv("bitcoin_data_features2.tsv", index=False, sep="\t")
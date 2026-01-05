import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

"""

這個檔案在做的事情：
1. 讀取合併後的比特幣資料集 (eth_data0.tsv)
2. 將標籤(label)進行分類，將相似的詐騙類型歸類到同一大類別中，共分為9大類別
3. 使用LabelEncoder將文字標籤轉換為數字編碼(label_idx)，以便於模型訓練
4. 計算每個類別的權重(class_weights)，以處理類別不平衡的問題
5. 繪製標籤分佈圖，視覺化各類別的數量分佈
6. 將處理後的數據儲存到新的TSV檔案(eth_data_processed.tsv)

"""

eth_data = pd.read_csv("eth_data0.tsv")


# 印出所有欄位名稱，看看它們到底長怎樣
eth_data["label"] = eth_data["label"].str.lower().str.strip()
print(eth_data['label'].unique())

def map_label_to_group(label):
    """
    將符合條件的label映射到專門類別內，總共將所有類別分類至9大類別內
    
    :param label: str, the original label from the dataset
    :return: str, the mapped group label
    """
    l = str(label).lower().strip()

    mapping = {
        'technical_attacks': [
            'hack scam', 'contract exploit scam', 'metamorphic contract', 
            'wallet drainer', 'sim swap scam'
        ],
        'contract_traps': [
            'honeypot scam', 'rug pull scam', 'fake project scam', 
            'nft airdrop scam', 'eth liquidity scam'
        ],
        'social_fraud': [
            'phishing scam', 'impersonation scam', 'romance scam', 
            'pigbutchering scam', 'giveaway scam', 'investment scam', 
            'fake returns scam', 'blackmail scam'
        ],
        'extortion': [
            'ransomware', 'sextortion scam'
        ],
        'benign': [
            'benign'
        ],
        'other': [
            'other'
        ]
    }


    for label, group in mapping.items():
        if l in group:
            return label
    return "other"


# 將 label 透過 map_label_to_group 函數進行轉換，變成六大類別
eth_data["label"] = eth_data["label"].apply(map_label_to_group) 
#print(eth_data["label"].value_counts())


# 將類別以數字進行編碼，方便放入模型內訓練
le = LabelEncoder()
eth_data["label_idx"] = le.fit_transform(eth_data["label"]) 


# 計算類別權重以處理不平衡資料
weights = compute_class_weight(class_weight="balanced", classes=np.unique(eth_data["label_idx"]), y=eth_data["label_idx"])
class_weights = {i : float(weights[i]) for i in range(len(weights))} 
print("Class Weights:", class_weights)

# 將類別數據整理成字典格式，並繪製類別分佈圖
category = eth_data["label"].value_counts()
category_dict = {}
for cat, num in zip(category.index, category):
    category_dict[cat] = num

plt.figure(figsize=(12,6))
sns.barplot(x=list(category_dict.keys()), y=list(category_dict.values()))
plt.xticks(rotation=45, ha='right')
plt.title("Distribution of Labels in eth Data")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# 儲存處理後的數據
eth_data.to_csv("eth_data_processed1.tsv", index=False, sep="\t")




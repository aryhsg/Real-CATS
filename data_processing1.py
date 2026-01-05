import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

"""

這個檔案在做的事情：
1. 讀取合併後的比特幣資料集 (bitcoin_data.tsv)
2. 將標籤(label)進行分類，將相似的詐騙類型歸類到同一大類別中，共分為9大類別
3. 使用LabelEncoder將文字標籤轉換為數字編碼(label_idx)，以便於模型訓練
4. 計算每個類別的權重(class_weights)，以處理類別不平衡的問題
5. 繪製標籤分佈圖，視覺化各類別的數量分佈
6. 將處理後的數據儲存到新的TSV檔案(bitcoin_data_processed.tsv)

"""

bitcoin_data = pd.read_csv("bitcoin_data.tsv")


# 印出所有欄位名稱，看看它們到底長怎樣
bitcoin_data["label"] = bitcoin_data["label"].str.lower().str.strip()


def map_label_to_group(label):
    """
    將符合條件的label映射到專門類別內，總共將所有類別分類至9大類別內
    
    :param label: str, the original label from the dataset
    :return: str, the mapped group label
    """
    l = str(label).lower().strip()

    mapping = {
        'extortion': ['blackmail scam', 'ransomware', 'sextortion scam', 'extortion', 'ddos attack', 'ransom'],
        'financial_scam': ['investment scam', 'giveaway scam', 'trust trading scam', 'romance scam', 'mining scam', 
                           'fake project scam', 'pigbutchering scam', 'donation scam', 'advanced fee scam', 
                           'rug pull scam', 'ponzi scheme', 'broker scam', 'recovery scam', 'airdrop scam', 
                           '419 scam', 'pump scam', 'deposit scam', 'inheritance scam', 'pyramid scheme', 
                           'exit scam', 'sweepstakes scam', 'illuminati scam'],
        'theft_hack': ['phishing scam', 'hack', 'theft', 'malware', 'clipper', 'contract exploit scam', 
                       'data breach exploitation', 'man in the middle attack', 'bruteforce attack', 'wallet drainer'],
        'infrastructure': ['bitcoin tumbler', 'exchange', 'cash app', 'online wallet', 'electrum wallet', 'gambling'],
        'market_crime': ['darknet market', 'terrorism', 'drug trafficking', 'money laundering', 'child abuse and exploitation'],
        'social_eng': ['impersonation scam', 'social media scam', 'social security scam', 'social engineer', 'phone scam', 'fake service'],
        'general_fraud': ['fake returns scam', 'bank scam', 'sim swap scam', 'loan scam', 'employment scam', 
                          'delivery scam', 'ebay scam', 'ssh scam', 'rental scam', 'craigslist scam', 
                          'fixed match scam', 'escrow scam'],
        'benign': ['benign']
    }

    for label, group in mapping.items():
        if l in group:
            return label
    return "other"


# 將 label 透過 map_label_to_group 函數進行轉換，變成九大類別
bitcoin_data["label"] = bitcoin_data["label"].apply(map_label_to_group) 
#print(bitcoin_data["label"].value_counts())


# 將類別以數字進行編碼，方便放入模型內訓練
le = LabelEncoder()
bitcoin_data["label_idx"] = le.fit_transform(bitcoin_data["label"]) 


# 計算類別權重以處理不平衡資料
weights = compute_class_weight(class_weight="balanced", classes=np.unique(bitcoin_data["label_idx"]), y=bitcoin_data["label_idx"])
class_weights = {i : float(weights[i]) for i in range(len(weights))} 


# 將類別數據整理成字典格式，並繪製類別分佈圖
category = bitcoin_data["label"].value_counts()
category_dict = {}
for cat, num in zip(category.index, category):
    category_dict[cat] = num

"""plt.figure(figsize=(12,6))
sns.barplot(x=list(category_dict.keys()), y=list(category_dict.values()))
plt.xticks(rotation=45, ha='right')
plt.title("Distribution of Labels in Bitcoin Data")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()"""


# 儲存處理後的數據
bitcoin_data.to_csv("bitcoin_data_processed1.tsv", index=False, sep="\t")




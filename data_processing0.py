import pandas as pd

"""

這個檔案在做的事情：
1. 讀取CB與BB的比特幣資料集 (CB.tsv, BB.tsv)
2. 為CB資料集添加標籤欄位(criminal=1)，BB資料集添加標籤欄位(criminal=0)
3. 合併兩個資料集成一個完整的比特幣資料集(bitcoin_data.tsv)

"""

cb = pd.read_csv("CB.tsv", sep="\t")
bb = pd.read_csv("BB.tsv", sep="\t")

cb["criminal"] = 1
bb["criminal"] = 0

bitcoin_data = pd.concat([cb, bb], axis=0, ignore_index=True)

print(len(cb), len(bb), len(bitcoin_data))
print(bb["address"].nunique()) # check unique addresses in BB
print(cb["address"].nunique()) # check unique addresses in CB

bitcoin_data.to_csv("bitcoin_data.tsv", index=False) #save the combined data

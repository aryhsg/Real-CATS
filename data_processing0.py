import pandas as pd

"""

這個檔案在做的事情：
1. 讀取CB與BB的比特幣資料集 (CB.tsv, BB.tsv)
2. 為CB資料集添加標籤欄位(criminal=1)，BB資料集添加標籤欄位(criminal=0)
3. 合併兩個資料集成一個完整的比特幣資料集(bitcoin_data.tsv)

"""

ce = pd.read_csv("CE.tsv", sep="\t")
be = pd.read_csv("BE.tsv", sep="\t")

ce["criminal"] = 1
be["criminal"] = 0

eth_data = pd.concat([ce, be], axis=0, ignore_index=True)

print(len(ce), len(be), len(eth_data))
print(be["address"].nunique()) # check unique addresses in BE
print(ce["address"].nunique()) # check unique addresses in CE

eth_data.to_csv("eth_data0.tsv", index=False) #save the combined data

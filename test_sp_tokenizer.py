import sentencepiece as spm
import sys
sp = spm.SentencePieceProcessor()

sp.load(sys.argv[-1]) # load spm.model

text = """
中央氣象局今早5時30分最新氣象資料顯示，第11號颱風海葵過去3小時中心在高雄沿海附近滯留打轉，受地形影響其強度已減弱為輕度颱風且暴風圈略為縮小，中心目前在高雄西北方，其暴風圈仍籠罩花蓮、台東、台中以南陸地及澎湖，風雨持續中。
"""
# encode: text => id
print(sp.encode_as_pieces(text))
print(sp.encode_as_ids(text))

from transformers import AutoTokenizer,PreTrainedTokenizer
text = """
中央氣象局今早5時30分最新氣象資料顯示，第11號颱風海葵過去3小時中心在高雄沿海附近滯留打轉，受地形影響其強度已減弱為輕度颱風且暴風圈略為縮小，中心目前在高雄西北方，其暴風圈仍籠罩花蓮、台東、台中以南陸地及澎湖，風雨持續中。
"""


def convert_to_token(tokenizer:PreTrainedTokenizer,text):
    return tokenizer.convert_ids_to_tokens(tokenizer(text,add_special_tokens=False)["input_ids"])

zh_wiki_bpe_tokenizer = AutoTokenizer.from_pretrained("zh_wiki_bpe_model_hf")
gpt2_chinese_tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Wenzhong-GPT2-110M")

print("zh-wiki bpe tokeinzer")
print(tokens:=convert_to_token(zh_wiki_bpe_tokenizer,text),f"\ntoken_count:{len(tokens)}")
print()
print("Wenzhong tokeinzer")
print(tokens:=convert_to_token(gpt2_chinese_tokenizer,text),f"\ntoken_count:{len(tokens)}")
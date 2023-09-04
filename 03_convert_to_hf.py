from transformers import LlamaTokenizerFast

chinese_sp_model_file = "zh_wiki_bpe_model_sp/zh_wiki_bpe.model"
# LlamaTokenizer can use spm model file directly
tokenizer = LlamaTokenizerFast(vocab_file=chinese_sp_model_file)
tokenizer.save_pretrained("zh_wiki_bpe_model_hf")
print("done")

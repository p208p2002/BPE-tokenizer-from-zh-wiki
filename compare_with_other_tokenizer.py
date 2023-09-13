from transformers import AutoTokenizer,PreTrainedTokenizer,LlamaTokenizer
from tokenizers import Tokenizer,AddedToken
from tokenizers.models import WordLevel

def _build_word_level_tokenizer(vocab_path):
    unk_token = "<unk>"
    bos_token = "<s>"
    eos_token = "</s>"

    word_lists = open(vocab_path, "r").read().strip().split("\n")
    word_lists = [x.split("\t")[0] for x in word_lists][:50000]

    word_model = WordLevel({unk_token:0},unk_token=unk_token)
    tokenizer = Tokenizer(word_model)
    add_tokes = [AddedToken(word,lstrip=True,rstrip=True,normalized=False) for word in word_lists]
    tokenizer.add_tokens(add_tokes)
    tokenizer.add_special_tokens([bos_token,eos_token,unk_token])
    return tokenizer

def _exec_word_level_tokenizer(tokenizer,text):
    decode_str = tokenizer.decode(token_ids:=tokenizer.encode(text).ids,skip_special_tokens=False)
    token_count = len(token_ids)
    unk_count = 0
    for idx in token_ids:
        if idx == 0:
            unk_count+=1
    return decode_str,token_count,unk_count

def convert_to_token(tokenizer:PreTrainedTokenizer,text):
    return " ".join(tokenizer.convert_ids_to_tokens(tokenizer(text,add_special_tokens=False)["input_ids"]))

if __name__ == "__main__":
    
    text = """
    尚-雅克·盧梭（法語：Jean-Jacques Rousseau，法語發音：[ʒɑ̃ ʒak ʁuso]；1712年6月28日—1778年7月2日）是啟蒙時代的法國與日內瓦哲學家、政治理論家、文學家和音樂家。
    盧梭的小說作品《愛彌兒》（Émile）是一篇關於全人公民教育的哲學論文，對康德影響甚大。其言情小說《新愛洛伊斯》對前浪漫主義（pre-romanticism）[19]及浪漫主義時期的小說發展十分重要[20]。
    不過，一些知名學者認為盧梭雖然預示了浪漫主義的誕生，但是其「現代文學姿態」其實早已「超越了感傷的浪漫主義」，而其嶄新的語言觀甚至「一直延續到了超現實主義那裡」[21]。
    
    The 1867 U.S. Senate election in Pennsylvania, voted on by the state legislature, was held on January 15, 1867. Simon Cameron was elected to the Senate for the third time; he had been chosen in 1845 and in 1857. Cameron and Governor Andrew Curtin each led a faction of Republicans and had clashed as early as 1855. Cameron tried to block Curtin from the party nomination for governor in 1860, while Curtin attempted to get Cameron excluded from Abraham Lincoln's cabinet; each failed.
    """
    
    #
    char_tokenizer = _build_word_level_tokenizer("data/char_vocab_29023.txt")
    word_tokenizer = _build_word_level_tokenizer("data/word_vocab_top_50000.txt")
    
    #
    zh_wiki_bpe_tokenizer = AutoTokenizer.from_pretrained("zh_en_wiki_bpe_model_hf")
    gpt2_chinese_tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Wenzhong-GPT2-110M")
    llama_tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    chinese_llama_tokenizer = LlamaTokenizer.from_pretrained("ziqingyang/chinese-llama-2-7b")
    linly_llama_tokenizer = LlamaTokenizer.from_pretrained("Linly-AI/Chinese-LLaMA-2-7B-hf")
    flagalpha_llama_tokenizer = LlamaTokenizer.from_pretrained("FlagAlpha/Llama2-Chinese-7b-Chat")
    
    

    print("zh-wiki bpe tokeinzer")
    print(token_str:=convert_to_token(zh_wiki_bpe_tokenizer,text),f"\ntoken_count:{len(token_str.split())}")
    
    print()
    print("Wenzhong tokeinzer")
    print(token_str:=convert_to_token(gpt2_chinese_tokenizer,text),f"\ntoken_count:{len(token_str.split())}")
    print()
    
    print("LLaMA (original)")
    print(token_str:=convert_to_token(llama_tokenizer,text),f"\ntoken_count:{len(token_str.split())}")
    print()
    
    print("Chinese LLaMA (ziqingyang/chinese-llama-2)")
    print(token_str:=convert_to_token(chinese_llama_tokenizer,text),f"\ntoken_count:{len(token_str.split())}")
    print()
    
    print("Chinese LLaMA (Linly-AI/Chinese-LLaMA-2-7B-hf)")
    print(token_str:=convert_to_token(linly_llama_tokenizer,text),f"\ntoken_count:{len(token_str.split())}")
    print()
    
    print("Chinese LLaMA (FlagAlpha/Llama2-Chinese-7b-Chat)")
    print(token_str:=convert_to_token(flagalpha_llama_tokenizer,text),f"\ntoken_count:{len(token_str.split())}")
    print()
    
    print("zh-word tokeinzer")
    decode_str,token_count,unk_count=_exec_word_level_tokenizer(word_tokenizer,text)
    print(f"{decode_str}\ntoken_count:{token_count}\nunk_count:{unk_count}")
    print()
    
    print("zh-char tokeinzer")
    decode_str,token_count,unk_count=_exec_word_level_tokenizer(char_tokenizer,text)
    print(f"{decode_str}\ntoken_count:{token_count}\nunk_count:{unk_count}")
    print()

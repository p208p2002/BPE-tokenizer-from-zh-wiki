from datasets import load_dataset
import os

dataset = iter(load_dataset("graelo/wikipedia", "20230601.zh",streaming=True,split="train"))
en_dataset = iter(load_dataset("graelo/wikipedia", "20230601.en",streaming=True,split="train"))
os.makedirs("data",exist_ok=True)

out_f = open("data/corpus.txt","w",buffering=3*1024*1024)
for i in range(2*10**6):
    if i % 2 == 0:
        data = next(dataset)
    else:
        data = next(en_dataset)
    text = data["text"]
    out_f.write(f"{text}\n")
    print(i,end="\r")

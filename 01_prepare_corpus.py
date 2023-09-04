from datasets import load_dataset
import os

dataset = load_dataset("graelo/wikipedia", "20230601.zh",streaming=True,split="train")
os.makedirs("data",exist_ok=True)
out_f = open("data/corpus.txt","w",buffering=3*1024*1024)
for idx,data in enumerate(dataset):
    text = data["text"]
    out_f.write(f"{text}\n")
    print(idx,end="\r")
        
import json
from transformers import AutoTokenizer
from tqdm import tqdm
j = json.load(open("/home/maruf/zarzis/dpr_deps/cpp_code_kng_base.json"))
tt = []
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
for d in tqdm(j):
    if len(tokenizer.encode(d["text"])) < 501:
            tt.append(d)

print(len(tt))
with open("tmp", "w") as wp:
    json.dump(tt, wp)
 
# https://engineering.linecorp.com/ja/blog/3.6-billion-parameter-japanese-language-model

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

device = 0 if torch.cuda.is_available() else torch.device('cpu')

print("使用するデバイス")
print(device)

print("----------------")
print("モデルの読み込み中")
model = AutoModelForCausalLM.from_pretrained(
    "line-corporation/japanese-large-lm-3.6b", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(
    "line-corporation/japanese-large-lm-3.6b", use_fast=False)
generator = pipeline("text-generation", model=model,
                     tokenizer=tokenizer, device=device)
set_seed(101)

print("----------------")

exitFlag = "0"

while exitFlag == "0":
    print("Prompt?")
    prompt = input()

    while prompt != "":
        text = generator(
            prompt,
            max_length=len(prompt) + 150,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1,
        )

        for t in text:
            prompt = t["generated_text"]
            print(t["generated_text"])

        print("この文章を続けますか？ 1 = YES, 0 = NO")
        continueFlag = input()
        if continueFlag == "0":
            prompt = ""

    print("プログラムを終了しますか？ 1 = YES, 0 = NO")
    exitFlag = input()

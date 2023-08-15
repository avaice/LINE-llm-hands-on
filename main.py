import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

device = 0 if torch.cuda.is_available() else torch.device('cpu') 
  
model = AutoModelForCausalLM.from_pretrained("line-corporation/japanese-large-lm-3.6b", torch_dtype=torch.float32)
# float16は指定しなくても問題ありません
tokenizer = AutoTokenizer.from_pretrained("line-corporation/japanese-large-lm-3.6b", use_fast=False)
# use_fast=False は必ず付与してください。なくても動きますが、我々の学習状況とは異なるので性能が下がります。
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
set_seed(101)
 
text = generator(
    "おはようございます、今日の天気は",
    max_length=30,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    num_return_sequences=5,
)
 
for t in text:
  print(t)
import torch
from transformers import PhiForCausalLM
from transformers import AutoTokenizer

torch.set_default_device("cuda")
model = PhiForCausalLM.from_pretrained("/root/autodl-tmp/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/phi-2", trust_remote_code=True)
inputs = tokenizer('Hello? How are u?', return_tensors="pt", return_attention_mask=False)
print(inputs)
embeddings = model.module.embd(inputs)
outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)

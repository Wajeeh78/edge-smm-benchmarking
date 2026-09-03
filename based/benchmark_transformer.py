import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("mps")
# Standard 124M parameter Transformer to match Mamba's 130M scale
model_id = "gpt2" 

print("Loading standard Transformer into M4 memory...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# GPT-2 does not use a pad token by default, so we set it to the EOS token
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

prompt = "The future of efficient machine learning relies on structured state space models because"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

print("\nGenerating response and benchmarking Transformer throughput...")

start_time = time.time()

with torch.no_grad():
    output_ids = model.generate(
        input_ids, 
        max_length=100, 
        temperature=0.7, 
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
end_time = time.time()

generated_tokens = len(output_ids[0]) - len(input_ids[0])
time_taken = end_time - start_time
tokens_per_sec = generated_tokens / time_taken

print(f"\n[Comparative Baseline] Hardware: M4 (MPS) | Architecture: Transformer | Throughput: {tokens_per_sec:.2f} tokens/second")
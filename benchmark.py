import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("mps")
model_id = "state-spaces/mamba-130m-hf"

print("Loading model and tokenizer into M4 memory...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

# The research prompt
prompt = "The future of efficient machine learning relies on structured state space models because"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

print("\nGenerating response and benchmarking throughput...")

# Start the timer
start_time = time.time()

# Generate the sequence without tracking gradients (saves memory)
with torch.no_grad():
    output_ids = model.generate(
        input_ids, 
        max_length=100, 
        temperature=0.7, 
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
# Stop the timer
end_time = time.time()

# Calculate performance metrics
generated_tokens = len(output_ids[0]) - len(input_ids[0])
time_taken = end_time - start_time
tokens_per_sec = generated_tokens / time_taken

# Decode the output
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\n--- Model Output ---")
print(response)
print("--------------------")
print(f"\n[Research Baseline] Hardware: M4 (MPS) | Throughput: {tokens_per_sec:.2f} tokens/second")
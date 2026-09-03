import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("mps")

def run_benchmark(model_name, model_id, lengths, is_transformer=False):
    print(f"\n--- Profiling Architecture: {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if is_transformer:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load model into unified memory
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    
    for length in lengths:
        # Create a dummy input sequence of exactly 'length' tokens
        input_ids = torch.randint(10, 1000, (1, length)).to(device)
        
        # Warm-up run (wakes up the Apple Silicon GPU)
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
            
        # Actual timed run: Generating 20 tokens based on the input context
        start_time = time.time()
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
        end_time = time.time()
        
        time_taken = end_time - start_time
        print(f"Input Context: {length:4d} tokens | Time to generate: {time_taken:.3f} seconds")
        
    # Clear the model from unified memory to ensure a fair test for the next architecture
    del model
    torch.mps.empty_cache()

# The sequence lengths we want to test
lengths_to_test = [100, 500, 1000, 2000]

# 1. Test the State Space Model
run_benchmark("Mamba-130M (State Space)", "state-spaces/mamba-130m-hf", lengths_to_test, is_transformer=False)

# 2. Test the Transformer
run_benchmark("GPT-2 (Transformer)", "gpt2", lengths_to_test, is_transformer=True)
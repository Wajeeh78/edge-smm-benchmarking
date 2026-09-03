import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Target the M4 chip
device = torch.device("mps")
print(f"Targeting hardware: {device}")

# 2. Define the State Space Model (130 million parameters for our initial baseline)
model_id = "state-spaces/mamba-130m-hf"

print(f"\nDownloading tokenizer and {model_id} architecture...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. Load the model in half-precision (float16) for extreme efficiency on Apple Silicon
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

# 4. Push the massive parameter matrix onto the Mac's unified memory
model.to(device)

print("\nSUCCESS! The State Space Model is completely loaded into the M4's memory.")
print(f"Current Model Location: {model.device}")
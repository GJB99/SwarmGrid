import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import os
import time

VISION_MODEL = "unsloth/gemma-3n-e4b-it-unsloth-bnb-4bit"
ADAPTER_PATH = "./models/finetuned_gemma_warehouse"

def diag():
    print(f"--- Vision Model Diagnostic Start ---")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Initial VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
    
    try:
        print(f"\n1. Loading Base Model: {VISION_MODEL}")
        model = AutoModelForImageTextToText.from_pretrained(
            VISION_MODEL,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print(f"Base Model Loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f}GB")
        
        if os.path.exists(ADAPTER_PATH):
            print(f"\n2. Loading Adapters: {ADAPTER_PATH}")
            model = PeftModel.from_pretrained(model, ADAPTER_PATH)
            print("Adapters Merged.")
            print(f"Final VRAM (Vision): {torch.cuda.memory_allocated(0)/1e9:.2f}GB")
        else:
            print(f"Adapter path not found: {ADAPTER_PATH}")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    diag()

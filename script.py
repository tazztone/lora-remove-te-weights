import safetensors.torch
import os

# --- Configuration ---
# 1. Path to your original LoRA file
input_lora_path = "m1r4.safetensors"

# 2. Path for the new, U-Net only LoRA file
output_lora_path = "m1r4_unet_only.safetensors"

# 3. This is the correct prefix for the main model weights you want to keep
UNET_PREFIX = "lora_unet_"

# --- Script ---
if not os.path.exists(input_lora_path):
    print(f"Error: Input file not found at '{input_lora_path}'")
else:
    print(f"Loading LoRA from: {input_lora_path}")
    state_dict = safetensors.torch.load_file(input_lora_path)
    
    filtered_state_dict = {}
    
    print(f"Filtering to keep only keys starting with '{UNET_PREFIX}'...")
    for key, value in state_dict.items():
        if key.startswith(UNET_PREFIX):
            filtered_state_dict[key] = value
            
    if not filtered_state_dict:
        print("\nError: No U-Net keys were found with that prefix.")
        print("Please double-check the UNET_PREFIX variable in the script.")
    else:
        # Save the filtered dictionary to a new file
        safetensors.torch.save_file(filtered_state_dict, output_lora_path)
        print(f"\nSuccess! New LoRA saved to: {output_lora_path}")
        print(f"{len(filtered_state_dict)} U-Net keys were kept out of {len(state_dict)} total keys.")
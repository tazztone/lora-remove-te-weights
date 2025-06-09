# LoRA Remove Text Encoder Weights

This script filters a LoRA `.safetensors` file to keep only the U-Net weights, removing any text encoder weights. Useful for sharing or optimizing LoRA files for inference.

## Files
- `script.py`: Main script to filter LoRA weights.
- `inspect-lora.py`: (Optional) Script to inspect LoRA keys.
- `lora_keys.txt`: (Optional) Example output of LoRA keys.

## Usage
1. Place your LoRA file (e.g., `m1r4.safetensors`) in this directory.
2. Edit `script.py` to set the correct input/output filenames if needed.
3. (Recommended) Use a Python virtual environment.
4. Install dependencies:
   ```pwsh
   pip install torch safetensors numpy
   ```
5. Run the script:
   ```pwsh
   python script.py
   ```

## Output
- A new file (e.g., `m1r4_unet_only.safetensors`) containing only U-Net weights.

## Notes
- Do **not** commit large model files (`.safetensors`) to GitHub.
- Add your virtual environment and model files to `.gitignore` (see below).

## License
MIT

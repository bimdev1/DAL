#!/usr/bin/env python3
"""Download a small GGUF model for testing."""

from huggingface_hub import hf_hub_download
import os

def download_gguf_model():
    """Download a small GGUF model for testing."""
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download a small GGUF model (TinyLlama 1.1B)
    print("Downloading TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf...")
    model_path = hf_hub_download(
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        local_dir="./models",
        local_dir_use_symlinks=False
    )
    
    print(f"Model downloaded to: {model_path}")
    return model_path

if __name__ == "__main__":
    download_gguf_model()

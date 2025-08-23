#!/usr/bin/env python3
"""Test the local model loader with different model backends."""

import sys
import logging
from pathlib import Path
from dal.model_loader import test_local_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model(model_spec, prompt):
    """Test a model with the given spec and prompt."""
    model_name = model_spec.get('model_name_or_path', 'unknown')
    print(f"\n{'='*80}")
    print(f"Testing model: {model_name}")
    print(f"Backend: {model_spec['backend']}")
    print(f"Prompt: {prompt}")
    
    result = test_local_model(model_spec, prompt)
    
    if result["success"]:
        print("\n✅ Test successful!")
        print(f"Model info: {result['model_info']}")
        print(f"Generated text: {result['generation_result']['output_text']}")
        print(f"Tokens: {result['generation_result']['tokens_prompt']} in, "
              f"{result['generation_result']['tokens_generated']} out, "
              f"{result['generation_result']['time_seconds']:.2f}s")
        return True
    else:
        print("\n❌ Test failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")
        return False

def main():
    test_prompt = "Hello, how are you today?"
    
    # Test with Hugging Face model
    hf_success = test_model(
        {
            "backend": "huggingface",
            "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "device_map": "auto",
            "load_in_4bit": True,
        },
        test_prompt
    )
    
    # Test with GGUF model if available
    gguf_path = Path("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    if gguf_path.exists():
        gguf_success = test_model(
            {
                "backend": "gguf",
                "model_name_or_path": str(gguf_path),
                "n_threads": 4,
                "n_ctx": 2048,
            },
            test_prompt
        )
    else:
        print(f"\n⚠️ GGUF model not found at {gguf_path}. Skipping GGUF test.")
        print("Run download_gguf_model.py first to download a test model.")
        gguf_success = False
    
    # Print summary
    print("\n" + "="*80)
    print("Test Summary:")
    print(f"✅ Hugging Face: {'PASSED' if hf_success else 'FAILED'}")
    print(f"✅ GGUF: {'PASSED' if gguf_success and gguf_path.exists() else 'SKIPPED' if not gguf_path.exists() else 'FAILED'}")
    
    # Exit with error code if any test failed
    if not (hf_success and (not gguf_path.exists() or gguf_success)):
        sys.exit(1)

if __name__ == "__main__":
    main()

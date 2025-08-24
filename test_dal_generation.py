"""Test script for DAL v3 local generation."""

import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from dal.local_generator import LocalGenerator
from dal.dal_types import DALTagBlock

def test_dal_generation():
    """Test DAL v3 local generation with a simple prompt."""
    print("Testing DAL v3 local generation...")
    
    # Model spec - same as in dal_runner.py
    model_spec = {
        "backend": "gguf",
        "model_name_or_path": "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "n_ctx": 2048,
        "n_threads": 4,
        "n_gpu_layers": 0  # CPU only
    }
    
    # Initialize generator
    print("Initializing generator...")
    generator = LocalGenerator(model_spec)
    
    if not generator.is_ready():
        print("ERROR: Failed to initialize generator")
        return
    
    # Test prompt
    test_prompt = "Explain how a computer works in simple terms."
    
    # Create SDT
    sdt = DALTagBlock()
    sdt.attributes = {
        "tone": "informative",
        "depth": "basic",
        "format": "concise",
        "style": "narrative"
    }
    
    # Generate response
    print("\nGenerating response...")
    try:
        response = generator.generate(
            text=test_prompt,
            sdt=sdt,
            max_new_tokens=100,
            temperature=0.7
        )
        
        print("\n=== GENERATION RESULT ===")
        print(f"Prompt: {test_prompt}")
        print(f"Response: {response.get('text', 'No response generated')}")
        print(f"Tokens generated: {response.get('tokens_out', 0)}")
        print("=======================")
        
    except Exception as e:
        print(f"\nERROR during generation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dal_generation()

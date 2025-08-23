"""Runner for GPT-2 model in benchmark tests."""

from typing import Optional
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Global model and tokenizer
_model = None
_tokenizer = None

def load_gpt2():
    """Load the GPT-2 model and tokenizer."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print("Loading GPT-2 model...")
        _tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        _model = GPT2LMHeadModel.from_pretrained("gpt2")
        _model.eval()
        print("GPT-2 model loaded.")
    return _model, _tokenizer

def generate_gpt2_response(prompt: str, max_tokens: int = 100) -> str:
    """Generate a response using GPT-2.
    
    Args:
        prompt: The input prompt
        max_tokens: Maximum number of tokens in the response
        
    Returns:
        Generated response text
    """
    model, tokenizer = load_gpt2()
    
    # Encode the input
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + max_tokens,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and clean up the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the input prompt from the response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response

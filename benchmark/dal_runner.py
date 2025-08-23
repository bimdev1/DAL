"""Runner for DAL model in benchmark tests using v3 pipeline with local generation."""

import os
from pathlib import Path
import sys
from typing import Dict, Any, Optional

# Add parent directory to path to import DAL
sys.path.append(str(Path(__file__).parent.parent))

from dal.pipeline_v3 import run_enhanced_pipeline_v3, PipelineConfig
from dal.local_generator import LocalGenerator

def generate_dal_response(prompt: str, max_tokens: int = 100, domain: Optional[str] = None) -> Dict[str, Any]:
    """Generate a response using the DAL v3 pipeline with local generation.
    
    Args:
        prompt: The input prompt
        max_tokens: Target token count (approximate)
        domain: Optional domain for SDT context
        
    Returns:
        Dictionary containing response and metadata
    """
    try:
        # Configure local generation
        model_spec = {
            "backend": "gguf",
            "model_name_or_path": "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "n_ctx": 2048,
            "n_threads": 4,
            "n_gpu_layers": 0  # Set to 0 for CPU-only
        }
        
        # Create local generator
        local_generator = LocalGenerator(model_spec)
        
        # Create SDT context if domain is provided
        from dal.dal_types import DALTagBlock
        sdt = DALTagBlock()
        if domain:
            sdt.attributes = {
                "tone": "informative",
                "depth": "detailed",
                "format": "concise",
                "style": "academic" if domain.lower() == "science" else "narrative"
            }
        
        # Generate response
        result = local_generator.generate(
            text=prompt,
            sdt=sdt,
            max_new_tokens=max_tokens,
            temperature=0.7,
            stop=["\n", "###"]
        )
        
        # Format the response
        return {
            "response": result["text"],
            "token_count": result.get("tokens_out", 0),
            "metrics": {
                "tokens_in": result.get("tokens_in", 0),
                "tokens_out": result.get("tokens_out", 0),
                "duration": result.get("duration", 0.0),
                "backend": result.get("backend", "unknown"),
                "model": result.get("model", "unknown")
            },
            "matched_keywords": [],  # Will be filled in by evaluate_factual_synthesis
            "keyword_match_ratio": 0.0  # Will be calculated by evaluate_factual_synthesis
        }
        
    except Exception as e:
        print(f"Error in generate_dal_response: {str(e)}")
        return {
            "response": f"An error occurred: {str(e)}",
            "token_count": 0,
            "model": "dal_v3",
            "error": str(e),
            "matched_keywords": [],
            "keyword_match_ratio": 0.0
        }

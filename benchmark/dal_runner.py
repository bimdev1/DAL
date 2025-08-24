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
        Dictionary containing response and metadata with required fields:
        - response: Generated text
        - token_count: Number of tokens in response
        - metrics: Dictionary of generation metrics
        - success: Boolean indicating success
        - model: Model name
        - matched_keywords: List of matched keywords
        - keyword_match_ratio: Ratio of matched keywords to total expected keywords
    """
    # Initialize result with default values
    result = {
        'response': '',
        'token_count': 0,
        'metrics': {},
        'success': False,
        'model': 'dal_v3',
        'matched_keywords': [],
        'keyword_match_ratio': 0.0
    }
    
    try:
        print(f"\n{'='*80}")
        print(f"GENERATING RESPONSE FOR PROMPT: {prompt[:200]}...")
        print(f"DOMAIN: {domain}")
        print(f"MAX TOKENS: {max_tokens}")
        
        # Configure local generation
        model_spec = {
            "backend": "gguf",
            "model_name_or_path": "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "n_ctx": 2048,
            "n_threads": 4,
            "n_gpu_layers": 0,  # Set to 0 for CPU-only
            "verbose": True     # Enable verbose output
        }
        
        print("Creating LocalGenerator with model spec:", model_spec)
        local_generator = LocalGenerator(model_spec)
        
        # Create SDT context if domain is provided
        from dal.dal_types import DALTagBlock
        from dal.sdt_injector import SDTOpts
        
        sdt = DALTagBlock()
        if domain:
            sdt.attributes = {
                "tone": "informative",
                "depth": "detailed",
                "format": "concise",
                "style": "academic" if domain.lower() == "science" else "narrative"
            }
            print(f"Using SDT attributes: {sdt.attributes}")
        
        # Configure pipeline with explicit model settings and SDT options
        sdt_opts = {
            'prompt_style': 'llama_inst',
            'include_constraints': True,
            'extra_constraints': ["Be factual and concise"]
        }
        
        print("Creating PipelineConfig with max_new_tokens:", max_tokens)
        config = PipelineConfig(
            use_local_generation=True,
            use_reflexive=True,
            reflexive_allow_mini_gen=True,
            sdt_control=True,
            force_regen=True,  # Force regeneration of all segments
            max_new_tokens=max_tokens,
            local_model_spec=model_spec,
            sdt_opts=sdt_opts
        )
        
        print("PipelineConfig created successfully")
        
        # Generate response using the pipeline
        print(f"\n{'='*80}")
        print("CALLING run_enhanced_pipeline_v3")
        print(f"Prompt: {prompt[:200]}...")
        print("Config:", {k: v for k, v in vars(config).items() if k != 'sdt_opts'})
        
        try:
            # Add debug print before calling the pipeline
            print("\n" + "="*40 + " START PIPELINE " + "="*40)
            pipeline_result = run_enhanced_pipeline_v3(prompt, config=config)
            print("\n" + "="*40 + " PIPELINE COMPLETE " + "="*40)
            
            print("Pipeline result keys:", list(pipeline_result.keys()) if pipeline_result else "None")
            
            # Extract response text and metrics
            response_text = pipeline_result.get('text', '')
            token_count = len(response_text.split()) if response_text else 0
            
            # Update result with pipeline response
            result.update({
                'response': response_text,
                'token_count': token_count,
                'metrics': {
                    'pipeline_metrics': pipeline_result.get('metrics', {}),
                    'sdt_applied': pipeline_result.get('sdt_applied', {})
                },
                'success': True
            })
            
            print(f"\nGenerated {token_count} tokens:")
            print("-"*80)
            print(response_text[:500] + ("..." if len(response_text) > 500 else ""))
            print("-"*80)
            
        except Exception as e:
            error_msg = f"Error in pipeline execution: {str(e)}"
            print("\n!!! " + "*"*40 + " PIPELINE ERROR " + "*"*40)
            print(error_msg)
            import traceback
            traceback.print_exc()
            print("*"*100 + "\n")
            
            result.update({
                'response': error_msg,
                'success': False,
                'metrics': {'error': str(e)}
            })
        
        return result
        
    except Exception as e:
        return {
            'response': f"Error generating response: {str(e)}",
            'tokens': 0,
            'metrics': {'error': str(e)},
            'success': False,
            'token_count': 0,
            'model': 'dal_v3',
            'error': str(e),
            'matched_keywords': [],
            'keyword_match_ratio': 0.0
        }

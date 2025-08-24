"""Debug script for DAL v3 local generation."""

import logging
import sys
import os
from pathlib import Path
import traceback
from typing import Dict, Any

# Enable debug logging
os.environ['DAL_DEBUG'] = '1'
os.environ['LLAMA_CPP_LOG_LEVEL'] = 'DEBUG'

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)30s - %(levelname)8s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_local_generator():
    """Test the LocalGenerator directly."""
    from dal.local_generator import LocalGenerator
    from dal.dal_types import DALTagBlock
    
    logger = logging.getLogger("test_local_generator")
    logger.info("="*80)
    logger.info("TESTING LOCAL GENERATOR DIRECTLY")
    logger.info("="*80)
    
    # Model spec - explicitly set device to CPU and disable GPU
    model_spec = {
        "backend": "gguf",
        "model_name_or_path": "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "n_ctx": 2048,
        "n_threads": 4,
        "n_gpu_layers": 0,
        "device": "cpu",
        "use_gpu": False
    }
    
    # Initialize generator
    logger.info("Initializing LocalGenerator...")
    try:
        generator = LocalGenerator(model_spec)
        if not generator.is_ready():
            logger.error("Generator is not ready after initialization")
            return False
        
        logger.info("Generator initialized successfully")
        
        # Create test prompt and SDT
        test_prompt = "Explain how a computer works in simple terms."
        sdt = DALTagBlock()
        sdt.attributes = {
            "tone": "informative",
            "depth": "basic",
            "format": "concise"
        }
        
        # Generate response
        logger.info("Generating response...")
        response = generator.generate(
            text=test_prompt,
            sdt=sdt,
            max_new_tokens=100,
            temperature=0.7
        )
        
        logger.info("="*40)
        logger.info("GENERATION RESULT")
        logger.info("="*40)
        logger.info(f"Response: {response.get('text', 'No response')}")
        logger.info(f"Tokens out: {response.get('tokens_out', 0)}")
        logger.info(f"Duration: {response.get('duration', 0):.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in test_local_generator: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_pipeline():
    """Test the full pipeline."""
    from dal.pipeline_v3 import run_enhanced_pipeline_v3, PipelineConfig
    
    logger = logging.getLogger("test_pipeline")
    logger.info("="*80)
    logger.info("TESTING PIPELINE")
    logger.info("="*80)
    
    try:
        # Configure pipeline with SDT options as a dictionary
        sdt_opts = {
            'prompt_style': 'llama_inst',
            'include_constraints': True,
            'extra_constraints': ["Be factual and concise"]
        }
        
        # Configure pipeline with explicit model settings
        config = PipelineConfig(
            use_local_generation=True,
            force_regen=True,
            max_new_tokens=100,
            sdt_control=True,
            sdt_opts=sdt_opts,
            use_reflexive=False,  # Disable reflexive optimization
            local_model_spec={
                'backend': 'gguf',
                'model_name_or_path': 'models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
                'n_ctx': 2048,
                'n_threads': 4,
                'n_gpu_layers': 0,
                'device': 'cpu',
                'use_gpu': False
            }
        )
        
        # Test prompt
        test_prompt = "Explain how a computer works in simple terms."
        logger.info(f"Test prompt: {test_prompt}")
        
        # Run pipeline
        logger.info("Running pipeline...")
        result = run_enhanced_pipeline_v3(test_prompt, config=config)
        
        logger.info("="*40)
        logger.info("PIPELINE RESULT")
        logger.info("="*40)
        logger.info(f"Generated text: {result.get('text', 'No text generated')}")
        logger.info(f"Segments: {len(result.get('segments', []))}")
        
        if 'metrics' in result:
            metrics = result['metrics']
            logger.info("Generation metrics:")
            logger.info(f"  Segments processed: {metrics.get('generation', {}).get('segments_processed', 0)}")
            logger.info(f"  Segments regenerated: {metrics.get('generation', {}).get('segments_regenerated', 0)}")
            logger.info(f"  Total tokens out: {metrics.get('generation', {}).get('total_tokens_out', 0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in test_pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("Starting DAL v3 debug session...\n")
    
    # Test 1: Direct LocalGenerator test
    print("\n" + "="*80)
    print("TEST 1: LOCAL GENERATOR DIRECT TEST")
    print("="*80)
    if test_local_generator():
        print("\n✅ LocalGenerator test completed successfully")
    else:
        print("\n❌ LocalGenerator test failed")
    
    # Test 2: Full pipeline test
    print("\n" + "="*80)
    print("TEST 2: PIPELINE INTEGRATION TEST")
    print("="*80)
    if test_pipeline():
        print("\n✅ Pipeline test completed successfully")
    else:
        print("\n❌ Pipeline test failed")
    
    print("\nDebug session complete.")

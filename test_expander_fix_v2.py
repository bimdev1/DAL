#!/usr/bin/env python3
"""Improved test script for the fixed ExpanderV2 implementation."""

import logging
import time
from typing import Tuple
from dal.expander_v2_fixed import ExpanderV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def measure_expansion(expander, text: str) -> Tuple[str, float, int, int]:
    """Run expansion and measure performance."""
    start_time = time.time()
    result = expander.expand(text)
    elapsed = time.time() - start_time
    
    input_words = len(text.split())
    output_words = len(result.split())
    
    return result, elapsed, input_words, output_words

def test_expander():
    """Test the ExpanderV2 with different input lengths and types."""
    logger.info("Initializing ExpanderV2...")
    expander = ExpanderV2()
    
    test_cases = [
        # Very short (should pass through)
        ("What is quantum mechanics?", "short question"),
        ("Explain briefly.", "very short"),
        
        # Short but complete thought (might expand slightly)
        ("The cat sat on the mat.", "short sentence"),
        
        # Medium length (should expand)
        ("""Quantum mechanics is a fundamental theory in physics that provides """
         """a description of the physical properties of nature at the scale """
         """of atoms and subatomic particles.""", 
         "medium explanation"),
         
        # Long input (should compress)
        ("""Quantum mechanics is a fundamental theory in physics that provides """
         """a description of the physical properties of nature at the scale of """
         """atoms and subatomic particles. It is the foundation of all quantum """
         """physics including quantum chemistry, quantum field theory, quantum """
         """technology, and quantum information science. Classical physics, the """
         """description of physics that existed before theories of relativity """
         """and quantum mechanics, describes many aspects of nature at an """
         """ordinary (macroscopic) scale, while quantum mechanics explains the """
         """aspects of nature at small (atomic and subatomic) scales, for which """
         """classical mechanics is insufficient.""", 
         "long explanation"),
         
        # Question format (should handle gracefully)
        ("""What are the key differences between classical and quantum mechanics? """
         """Please explain in simple terms.""",
         "question format")
    ]
    
    print("\n" + "="*80)
    print("TESTING EXPANDER V2 IMPROVEMENTS")
    print("="*80)
    
    for i, (text, desc) in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {desc.upper()}")
        print(f"{'='*80}")
        print(f"INPUT ({len(text.split())} words):")
        print(f"  {text}")
        
        try:
            result, elapsed, in_words, out_words = measure_expansion(expander, text)
            ratio = out_words / in_words if in_words > 0 else 1.0
            
            print(f"\nOUTPUT ({out_words} words, {elapsed:.2f}s, {ratio:.1f}x):")
            print(f"  {result}")
            
            # Print analysis
            if in_words < 10 and in_words == out_words:
                print("✓ Short input passed through unchanged")
            elif in_words > 50 and ratio < 0.9:
                print(f"✓ Long input compressed ({ratio:.1f}x)")
            elif 10 <= in_words <= 50 and ratio > 1.2:
                print(f"✓ Medium input expanded ({ratio:.1f}x)")
                
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    test_expander()

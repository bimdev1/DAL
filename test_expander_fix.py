#!/usr/bin/env python3
"""Test script for the fixed ExpanderV2 implementation."""

import logging
from dal.expander_v2_fixed import ExpanderV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_expander():
    """Test the ExpanderV2 with different input lengths and types."""
    expander = ExpanderV2()
    
    test_cases = [
        # Very short (should pass through)
        ("What is quantum mechanics?", "short question"),
        ("Explain briefly.", "very short"),
        
        # Medium length (should expand)
        ("Quantum mechanics is a fundamental theory in physics that provides "
         "a description of the physical properties of nature at the scale "
         "of atoms and subatomic particles.", "medium explanation"),
         
        # Long input (should compress)
        ("""Quantum mechanics is a fundamental theory in physics that provides a description """
         """of the physical properties of nature at the scale of atoms and subatomic """
         """particles. It is the foundation of all quantum physics including quantum """
         """chemistry, quantum field theory, quantum technology, and quantum information """
         """science. Classical physics, the description of physics that existed before """
         """theories of relativity and quantum mechanics, describes many aspects of nature """
         """at an ordinary (macroscopic) scale, while quantum mechanics explains the """
         """aspects of nature at small (atomic and subatomic) scales, for which classical """
         """mechanics is insufficient.""", "long explanation")
    ]
    
    for i, (text, desc) in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {desc} ({len(text.split())} words)")
        print(f"{'='*80}")
        print(f"INPUT:  {text[:150]}{'...' if len(text) > 150 else ''}")
        
        try:
            result = expander.expand(text)
            print(f"OUTPUT: {result[:200]}{'...' if len(result) > 200 else ''}")
            print(f"LENGTH: {len(text)} chars → {len(result)} chars")
            print(f"WORDS:  {len(text.split())} → {len(result.split())} words")
        except Exception as e:
            print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    test_expander()

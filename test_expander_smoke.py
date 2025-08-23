#!/usr/bin/env python3
"""Smoke test for ExpanderV2 implementation."""

import logging
from dal.expander_v2_fixed import ExpanderV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_smoke_test():
    """Run a smoke test on ExpanderV2."""
    print("🚀 Starting ExpanderV2 Smoke Test")
    print("=" * 80)
    
    # Initialize expander
    print("\nInitializing ExpanderV2...")
    expander = ExpanderV2()
    
    # Test cases
    test_cases = [
        ("What is quantum mechanics?", "short_question"),
        ("Explain briefly.", "very_short"),
        ("The cat sat on the mat.", "short_sentence"),
        (
            "Quantum mechanics is a fundamental theory in physics that provides "
            "a description of the physical properties of nature at the scale "
            "of atoms and subatomic particles.",
            "medium_explanation"
        ),
        (
            "Quantum mechanics is a fundamental theory in physics that provides "
            "a description of the physical properties of nature at the scale of "
            "atoms and subatomic particles. It is the foundation of all quantum "
            "physics including quantum chemistry, quantum field theory, quantum "
            "technology, and quantum information science. Classical physics, the "
            "description of physics that existed before theories of relativity "
            "and quantum mechanics, describes many aspects of nature at an "
            "ordinary (macroscopic) scale, while quantum mechanics explains the "
            "aspects of nature at small (atomic and subatomic) scales, for which "
            "classical mechanics is insufficient.",
            "long_explanation"
        ),
        ("", "empty_string"),
        (" ".join(["word"] * 100), "repetitive_content")
    ]
    
    # Run tests
    results = []
    for i, (text, name) in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {name}")
        print("-" * 80)
        print(f"Input: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        try:
            # Run expansion
            output = expander.expand(text)
            
            # Calculate stats
            input_words = len(text.split())
            output_words = len(output.split())
            ratio = output_words / input_words if input_words > 0 else 0
            
            print(f"Output: {output[:100]}{'...' if len(output) > 100 else ''}")
            print(f"Words: {input_words} → {output_words} (x{ratio:.2f})")
            
            # Basic validation
            if not text.strip() and output.strip():
                raise ValueError("Empty input should produce empty output")
                
            if input_words < 10 and input_words != output_words:
                print("⚠️  Short input was modified (this may be expected in some cases)")
            
            results.append((name, True, ""))
            print("✅ PASSED")
            
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"❌ FAILED: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"\n✅ PASSED: {passed}/{total} tests")
    print(f"❌ FAILED: {total - passed}/{total} tests")
    
    # Print failed tests
    failed = [(name, error) for name, success, error in results if not success]
    if failed:
        print("\nFailed tests:")
        for i, (name, error) in enumerate(failed, 1):
            print(f"  {i}. {name}: {error}")
    
    return all(success for _, success, _ in results)

if __name__ == "__main__":
    success = run_smoke_test()
    exit(0 if success else 1)

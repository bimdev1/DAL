#!/usr/bin/env python3
"""Comprehensive test script for the fixed ExpanderV2 implementation."""

import logging
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Container for test results and metrics."""
    input_text: str
    output_text: str
    input_tokens: int
    output_tokens: int
    duration: float
    test_case: str
    passed: bool
    error: Optional[str] = None
    
    @property
    def ratio(self) -> float:
        return self.output_tokens / self.input_tokens if self.input_tokens > 0 else 1.0
    
    def to_dict(self) -> Dict:
        return {
            'test_case': self.test_case,
            'input_text': self.input_text,
            'output_text': self.output_text,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'ratio': self.ratio,
            'duration': self.duration,
            'passed': self.passed,
            'error': self.error
        }

class ExpanderTester:
    """Test harness for ExpanderV2 with comprehensive test cases."""
    
    def __init__(self):
        from dal.expander_v2_fixed import ExpanderV2
        self.expander = ExpanderV2()
        self.results: List[TestResult] = []
        
    def count_tokens(self, text: str) -> int:
        """Count tokens using the expander's tokenizer."""
        if not self.expander._tokenizer:
            return len(text.split())  # Fallback to word count
        return len(self.expander._tokenizer.tokenize(text))
    
    def run_test_case(self, text: str, test_case: str) -> TestResult:
        """Run a single test case and return results."""
        start_time = time.time()
        error = None
        
        try:
            # Run the expander
            output = self.expander.expand(text)
            
            # Measure tokens
            input_tokens = self.count_tokens(text)
            output_tokens = self.count_tokens(output)
            
            # Basic validation
            passed = self._validate_output(text, output, input_tokens, output_tokens)
            
        except Exception as e:
            output = ""
            input_tokens = self.count_tokens(text)
            output_tokens = 0
            passed = False
            error = str(e)
            logger.error("Test failed: %s", error, exc_info=True)
        
        # Create result
        result = TestResult(
            input_text=text,
            output_text=output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration=time.time() - start_time,
            test_case=test_case,
            passed=passed,
            error=error
        )
        
        self.results.append(result)
        return result
    
    def _validate_output(self, input_text: str, output: str, 
                        input_tokens: int, output_tokens: int) -> bool:
        """Validate the expander output."""
        # Empty output is only valid if input was empty
        if not output:
            return not input_text.strip()
            
        # For very short inputs, output should match input
        if input_tokens < 10:
            return output.strip() == input_text.strip()
            
        # For short inputs, output should be similar or slightly longer
        if input_tokens < 30:
            return output_tokens <= max(30, input_tokens * 1.5)
            
        # For medium inputs, allow some compression/expansion
        if input_tokens < 100:
            return 0.5 <= (output_tokens / input_tokens) <= 1.5
            
        # For long inputs, allow more compression
        return 0.3 <= (output_tokens / input_tokens) <= 1.2
    
    def print_summary(self):
        """Print a summary of test results."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print(f"\n✅ PASSED: {passed}/{total} tests")
        print(f"❌ FAILED: {total - passed}/{total} tests")
        
        # Print failed tests
        failed = [r for r in self.results if not r.passed]
        if failed:
            print("\nFailed tests:")
            for i, result in enumerate(failed, 1):
                print(f"\n{i}. {result.test_case}")
                print(f"   Input: {result.input_text[:100]}...")
                print(f"   Error: {result.error or 'Validation failed'}")
        
        # Print performance stats
        if self.results:
            avg_time = sum(r.duration for r in self.results) / len(self.results)
            print(f"\n⏱️  Average processing time: {avg_time:.2f}s")
    
    def save_results(self, path: str = "expander_test_results.json"):
        """Save test results to a JSON file."""
        results = [r.to_dict() for r in self.results]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n📊 Results saved to {path}")

def get_test_cases() -> List[Tuple[str, str]]:
    """Return a list of (text, test_case_name) tuples."""
    return [
        # Very short (should pass through)
        ("What is quantum mechanics?", "short_question"),
        ("Explain briefly.", "very_short"),
        
        # Short but complete thoughts
        ("The cat sat on the mat.", "short_sentence"),
        ("Please summarize the key points.", "short_request"),
        
        # Questions (should pass through)
        ("What are the main differences between classical and quantum mechanics?", "medium_question"),
        
        # Medium length (should expand slightly)
        ("""Quantum mechanics is a fundamental theory in physics that provides """
         """a description of the physical properties of nature at the scale """
         """of atoms and subatomic particles.""", 
         "medium_explanation"),
         
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
         "long_explanation"),
         
        # Edge cases
        ("", "empty_string"),
        (" ".join(["word"] * 200), "long_repetitive"),
    ]

def main():
    """Run the test suite."""
    print("🚀 Starting ExpanderV2 Test Suite")
    print("=" * 80)
    
    tester = ExpanderTester()
    test_cases = get_test_cases()
    
    for i, (text, test_case) in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}/{len(test_cases)}: {test_case}")
        print("-" * 80)
        print(f"Input: {text[:150]}{'...' if len(text) > 150 else ''}")
        
        result = tester.run_test_case(text, test_case)
        
        print(f"Output: {result.output_text[:150]}{'...' if len(result.output_text) > 150 else ''}")
        print(f"Tokens: {result.input_tokens} → {result.output_tokens} (x{result.ratio:.2f})")
        print(f"Time: {result.duration:.2f}s")
        print(f"Status: {'✅ PASS' if result.passed else '❌ FAIL'}")
    
    # Print summary and save results
    tester.print_summary()
    tester.save_results()
    
    # Exit with error code if any tests failed
    if any(not r.passed for r in tester.results):
        exit(1)

if __name__ == "__main__":
    main()

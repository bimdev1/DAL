#!/usr/bin/env python3
"""
Comprehensive test suite for ExpanderV2 implementation.

Tests various scenarios including:
- Short inputs (should pass through)
- Medium inputs (should expand slightly)
- Long inputs (should compress)
- Edge cases (empty string, very long, etc.)
- Error handling
"""

import unittest
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the ExpanderV2 implementation
from dal.expander_v2_fixed import ExpanderV2, determine_segments, determine_max_length


@dataclass
class TestCase:
    """Container for test case data and expected results."""
    name: str
    input_text: str
    min_output_ratio: float = 0.0
    max_output_ratio: float = float('inf')
    should_expand: bool = True
    expected_segments: Optional[int] = None
    description: str = ""


class TestExpanderV2(unittest.TestCase):    
    @classmethod
    def setUpClass(cls):
        """Initialize the expander once for all tests."""
        logger.info("Initializing ExpanderV2...")
        cls.expander = ExpanderV2()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using the expander's tokenizer."""
        if not self.expander._tokenizer:
            return len(text.split())  # Fallback to word count
        return len(self.expander._tokenizer.tokenize(text))
    
    def run_test_case(self, test_case: TestCase) -> Tuple[bool, str]:
        """Run a single test case and return (success, message)."""
        # Skip if expander failed to initialize
        if not hasattr(self, 'expander') or not self.expander._summariser:
            return False, "Expander not initialized"
        
        start_time = time.time()
        
        try:
            # Test segment determination
            tokens = self.count_tokens(test_case.input_text)
            segments = determine_segments(tokens)
            
            if test_case.expected_segments is not None:
                self.assertEqual(segments, test_case.expected_segments,
                               f"Expected {test_case.expected_segments} segments, got {segments}")
            
            # Test expansion
            output = self.expander.expand(test_case.input_text)
            output_tokens = self.count_tokens(output)
            ratio = output_tokens / tokens if tokens > 0 else 1.0
            
            # Check if expansion matches expectations
            if test_case.should_expand:
                self.assertGreaterEqual(ratio, test_case.min_output_ratio,
                                     f"Output too short: {ratio:.2f} < {test_case.min_output_ratio}")
                self.assertLessEqual(ratio, test_case.max_output_ratio,
                                  f"Output too long: {ratio:.2f} > {test_case.max_output_ratio}")
            
            # Basic output validation
            self.assertIsInstance(output, str)
            if test_case.input_text.strip() and not test_case.should_expand:
                self.assertEqual(output.strip(), test_case.input_text.strip())
            
            duration = time.time() - start_time
            return True, f"PASSED in {duration:.2f}s (tokens: {tokens}→{output_tokens}, ratio: {ratio:.2f}x)"
            
        except Exception as e:
            return False, f"FAILED: {str(e)}"
    
    def test_short_inputs(self):
        """Test very short inputs that should pass through unchanged."""
        cases = [
            TestCase(
                name="empty_string",
                input_text="",
                should_expand=False,
                description="Empty string should pass through unchanged"
            ),
            TestCase(
                name="single_word",
                input_text="Hello",
                should_expand=False,
                description="Single word should pass through unchanged"
            ),
            TestCase(
                name="short_question",
                input_text="What is quantum mechanics?",
                should_expand=False,
                description="Short questions should pass through unchanged"
            ),
            TestCase(
                name="short_command",
                input_text="Explain briefly.",
                should_expand=False,
                description="Short commands should pass through unchanged"
            )
        ]
        
        self._run_test_cases(cases)
    
    def test_medium_inputs(self):
        """Test medium-length inputs that should be expanded."""
        cases = [
            TestCase(
                name="short_paragraph",
                input_text=("Quantum mechanics is a fundamental theory in physics that provides "
                          "a description of the physical properties of nature at the scale of "
                          "atoms and subatomic particles."),
                min_output_ratio=0.8,
                max_output_ratio=1.5,
                expected_segments=1,
                description="Medium paragraphs should be expanded slightly"
            ),
            TestCase(
                name="multiple_sentences",
                input_text=("The cat sat on the mat. The dog barked loudly. "
                          "The bird flew away. The sun was shining brightly."),
                min_output_ratio=0.7,
                max_output_ratio=1.5,
                expected_segments=1,
                description="Multiple sentences should be combined and expanded"
            )
        ]
        
        self._run_test_cases(cases)
    
    def test_long_inputs(self):
        """Test long inputs that should be compressed."""
        long_text = ("Quantum mechanics is a fundamental theory in physics that provides "
                    "a description of the physical properties of nature at the scale of "
                    "atoms and subatomic particles. It is the foundation of all quantum "
                    "physics including quantum chemistry, quantum field theory, quantum "
                    "technology, and quantum information science. Classical physics, the "
                    "description of physics that existed before theories of relativity "
                    "and quantum mechanics, describes many aspects of nature at an "
                    "ordinary (macroscopic) scale, while quantum mechanics explains the "
                    "aspects of nature at small (atomic and subatomic) scales, for which "
                    "classical mechanics is insufficient.")
        
        cases = [
            TestCase(
                name="long_paragraph",
                input_text=long_text,
                min_output_ratio=0.3,
                max_output_ratio=0.9,
                expected_segments=2,
                description="Long paragraphs should be compressed"
            ),
            TestCase(
                name="repetitive_content",
                input_text=" ".join(["word"] * 100),
                min_output_ratio=0.1,
                max_output_ratio=0.5,
                expected_segments=2,
                description="Highly repetitive content should be compressed significantly"
            )
        ]
        
        self._run_test_cases(cases)
    
    def test_edge_cases(self):
        """Test various edge cases and error conditions."""
        cases = [
            TestCase(
                name="whitespace_only",
                input_text="   \n  \t  \n",
                should_expand=False,
                description="Whitespace-only input should return empty string"
            ),
            TestCase(
                name="unicode_characters",
                input_text="量子力学は素晴らしいです！ これはテストです。",
                min_output_ratio=0.5,
                max_output_ratio=1.5,
                description="Unicode characters should be handled correctly"
            ),
            TestCase(
                name="very_long_word",
                input_text="Pneumonoultramicroscopicsilicovolcanoconiosis " * 10,
                min_output_ratio=0.1,
                max_output_ratio=0.5,
                description="Very long words should not break tokenization"
            )
        ]
        
        self._run_test_cases(cases)
    
    def test_determine_segments(self):
        """Test the determine_segments function."""
        test_cases = [
            (0, 1),    # Edge case: 0 tokens
            (1, 1),    # Minimal input
            (100, 1),  # Below first threshold
            (500, 1),  # Just below threshold
            (600, 2),  # Cross first threshold
            (1000, 2), # Middle of range
            (1500, 3), # Cross second threshold
            (2000, 3),  # Above all thresholds
        ]
        
        for tokens, expected in test_cases:
            with self.subTest(tokens=tokens):
                result = determine_segments(tokens)
                self.assertEqual(result, expected, 
                               f"Expected {expected} segments for {tokens} tokens, got {result}")
    
    def _run_test_cases(self, test_cases: List[TestCase]):
        """Helper to run multiple test cases and report results."""
        for case in test_cases:
            with self.subTest(name=case.name):
                logger.info(f"\n{'='*80}\nTEST: {case.name}\n{'-'*80}")
                logger.info(f"Description: {case.description}")
                logger.info(f"Input: {case.input_text[:150]}{'...' if len(case.input_text) > 150 else ''}")
                
                success, message = self.run_test_case(case)
                logger.info(f"Result: {message}")
                
                if not success:
                    self.fail(f"Test '{case.name}' failed: {message}")
                
                # If we get here, the test passed
                output = self.expander.expand(case.input_text)
                logger.info(f"Output: {output[:150]}{'...' if len(output) > 150 else ''}")


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

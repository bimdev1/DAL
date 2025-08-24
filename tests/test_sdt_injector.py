"""Tests for the SDT injector module."""

import unittest
from unittest.mock import MagicMock, patch

from dal.sdt_injector import (
    build_prompt, 
    SDTApplied, 
    SDTOpts, 
    _normalize_tone, 
    _normalize_format,
    _normalize_depth
)
from dal.dal_types import DALTagBlock

class TestSDTInjector(unittest.TestCase):
    """Test cases for the SDT injector functionality."""
    
    def test_normalize_tone(self):
        """Test tone normalization."""
        self.assertEqual(_normalize_tone("professional"), "professional")
        self.assertEqual(_normalize_tone("casual"), "neutral")
        self.assertEqual(_normalize_tone("excited"), "enthusiastic")
        self.assertEqual(_normalize_tone("doubtful"), "skeptical")
        self.assertEqual(_normalize_tone("unknown"), "neutral")
        self.assertIsNone(_normalize_tone(""))
        self.assertIsNone(_normalize_tone(None))
    
    def test_normalize_format(self):
        """Test format normalization."""
        self.assertEqual(_normalize_format("exposition"), "exposition")
        self.assertEqual(_normalize_format("bullets"), "bullet_list")
        self.assertEqual(_normalize_format("numbered"), "steps")
        self.assertEqual(_normalize_format("table"), "tabular")
        self.assertEqual(_normalize_format("unknown"), "exposition")
        self.assertIsNone(_normalize_format(""))
        self.assertIsNone(_normalize_format(None))
    
    def test_normalize_depth(self):
        """Test depth normalization."""
        self.assertEqual(_normalize_depth(1), 1)
        self.assertEqual(_normalize_depth(3), 3)
        self.assertEqual(_normalize_depth(5), 5)
        self.assertEqual(_normalize_depth("1"), 1)
        self.assertEqual(_normalize_depth("brief"), 1)
        self.assertEqual(_normalize_depth("detailed"), 5)
        self.assertEqual(_normalize_depth("normal"), 3)
        self.assertIsNone(_normalize_depth(""))
        self.assertIsNone(_normalize_depth(None))
    
    def test_build_prompt_default(self):
        """Test building a prompt with default options."""
        text = "Test content"
        sdt = DALTagBlock()
        prompt, applied = build_prompt(text, sdt)
        
        self.assertEqual(prompt, text)  # No changes expected
        self.assertIsInstance(applied, SDTApplied)
        self.assertIsNone(applied.tone)
        self.assertIsNone(applied.depth)
        self.assertIsNone(applied.format)
    
    def test_build_prompt_with_tone(self):
        """Test building a prompt with tone specification."""
        text = "Test content"
        sdt = DALTagBlock(tone="professional")
        prompt, applied = build_prompt(text, sdt)
        
        self.assertIn("professional", prompt.lower())
        self.assertEqual(applied.tone, "professional")
    
    def test_build_prompt_with_format(self):
        """Test building a prompt with format specification."""
        text = "Test content"
        sdt = DALTagBlock(format="steps")
        prompt, applied = build_prompt(text, sdt)
        
        self.assertIn("numbered steps", prompt.lower())
        self.assertEqual(applied.format, "steps")
    
    def test_build_prompt_with_depth(self):
        """Test building a prompt with depth specification."""
        text = "Test content"
        sdt = DALTagBlock(depth=5)
        prompt, applied = build_prompt(text, sdt)
        
        self.assertIn("thorough and comprehensive", prompt.lower())
        self.assertEqual(applied.depth, 5)
    
    def test_build_prompt_with_constraints(self):
        """Test that constraints are included when enabled."""
        text = "Test content"
        sdt = DALTagBlock()
        opts = SDTOpts(
            include_constraints=True,
            extra_constraints=["Be extra concise"]
        )
        prompt, applied = build_prompt(text, sdt, opts)
        
        self.assertIn("avoid hallucinations", prompt.lower())
        self.assertIn("extra concise", prompt.lower())
        self.assertIn("extra concise", applied.constraints)
    
    def test_build_prompt_prompt_styles(self):
        """Test different prompt formatting styles."""
        text = "Test content"
        sdt = DALTagBlock(tone="professional")
        
        # Test LLaMA instruction style (default)
        prompt, _ = build_prompt(text, sdt, SDTOpts(prompt_style='llama_inst'))
        self.assertIn("[INST]", prompt)
        self.assertIn("[/INST]", prompt)
        
        # Test ChatML style
        prompt, _ = build_prompt(text, sdt, SDTOpts(prompt_style='chatml'))
        self.assertIn("<|im_start|>", prompt)
        self.assertIn("<|im_end|>", prompt)
        
        # Test plain style
        prompt, _ = build_prompt(text, sdt, SDTOpts(prompt_style='plain'))
        self.assertIn("guidelines", prompt.lower())
        self.assertNotIn("<|im_start|>", prompt)
        self.assertNotIn("[INST]", prompt)

if __name__ == '__main__':
    unittest.main()

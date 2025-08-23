"""Tests for the local_generator module."""

import unittest
from unittest.mock import MagicMock, patch, ANY

from dal.local_generator import LocalGenerator
from dal.dal_types import DALTagBlock

class TestLocalGenerator(unittest.TestCase):
    """Test cases for LocalGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_spec = {
            'backend': 'huggingface',
            'model_name_or_path': 'test-model',
            'load_in_4bit': True
        }
        self.mock_model = MagicMock()
        self.mock_model.generate.return_value = {
            'output_text': 'Generated text',
            'tokens_prompt': 10,
            'tokens_generated': 20
        }
        
        # Patch the model loader to return our mock model
        self.patcher = patch('dal.local_generator.load_local_model', 
                           return_value=self.mock_model)
        self.mock_load_model = self.patcher.start()
        
    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
    
    def test_init_success(self):
        """Test successful initialization of LocalGenerator."""
        generator = LocalGenerator(self.model_spec)
        self.mock_load_model.assert_called_once_with(self.model_spec)
        self.assertIsNotNone(generator.model)
    
    def test_init_failure(self):
        """Test initialization failure is handled gracefully."""
        self.mock_load_model.side_effect = Exception('Load failed')
        generator = LocalGenerator(self.model_spec)
        self.assertIsNone(generator.model)
    
    def test_is_ready(self):
        """Test the is_ready method."""
        generator = LocalGenerator(self.model_spec)
        self.assertTrue(generator.is_ready())
        
        generator.model = None
        self.assertFalse(generator.is_ready())
    
    def test_generate_success(self):
        """Test successful text generation."""
        generator = LocalGenerator(self.model_spec)
        sdt = DALTagBlock(voice='professional')
        
        result = generator.generate(
            text='Test input',
            sdt=sdt,
            max_new_tokens=50,
            temperature=0.7
        )
        
        self.assertEqual(result['text'], 'Generated text')
        self.assertEqual(result['tokens_in'], 10)
        self.assertEqual(result['tokens_out'], 20)
        self.mock_model.generate.assert_called_once()
    
    @patch('dal.local_generator.load_local_model')
    def test_generate_not_ready(self, mock_load_model):
        """Test generation when model is not ready."""
        # Setup mock to fail loading
        mock_load_model.side_effect = Exception("Load failed")
        
        # Initialize and test
        generator = LocalGenerator(self.model_spec)
        
        test_text = "Test input"
        result = generator.generate(
            text=test_text,
            sdt=DALTagBlock(),
            max_new_tokens=50
        )
        
        # Should return the original text with error info
        self.assertEqual(result["text"], test_text)
        self.assertEqual(result["backend"], "passthrough")
        self.assertIn("error", result)
        self.assertIn("Model not loaded", result.get("error", ""))

if __name__ == "__main__":
    unittest.main()

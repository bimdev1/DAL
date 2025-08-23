"""Tests for the model_loader module."""

import unittest
from unittest.mock import patch, MagicMock
import torch
from dal.model_loader import (
    ModelBackend,
    ModelLoadError,
    load_local_model,
    test_local_model,
)

class TestModelLoader(unittest.TestCase):
    """Test cases for model_loader module."""

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_huggingface_model(self, mock_tokenizer, mock_model):
        """Test loading a Hugging Face model."""
        # Mock the model and tokenizer
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        # Test loading with default settings
        model = load_local_model({
            "backend": "huggingface",
            "model_name_or_path": "microsoft/phi-2"
        })
        
        self.assertEqual(model.backend, ModelBackend.HUGGINGFACE)
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()

    @patch('llama_cpp.Llama')
    def test_load_gguf_model(self, mock_llama):
        """Test loading a GGUF model."""
        # Mock the GGUF model
        mock_llama.return_value = MagicMock()
        
        # Test loading with custom settings
        model = load_local_model({
            "backend": "gguf",
            "model_name_or_path": "/path/to/phi-2.gguf",
            "n_threads": 4,
            "n_ctx": 2048
        })
        
        self.assertEqual(model.backend, ModelBackend.GGUF)
        mock_llama.assert_called_once()

    def test_invalid_backend(self):
        """Test loading with an invalid backend."""
        with self.assertRaises(ValueError):
            load_local_model({
                "backend": "invalid_backend",
                "model_name_or_path": "test"
            })

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_test_local_model_success(self, mock_tokenizer, mock_model):
        """Test the test_local_model function with a successful run."""
        # Mock the model and tokenizer
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        # Mock model generation
        mock_model.return_value.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.return_value.decode.return_value = "Test output"
        
        # Run the test
        result = test_local_model({
            "backend": "huggingface",
            "model_name_or_path": "microsoft/phi-2"
        })
        
        self.assertTrue(result["success"])
        self.assertIn("generation_result", result)
        self.assertEqual(result["generation_result"]["output_text"], "Test output")

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_test_local_model_failure(self, mock_model):
        """Test the test_local_model function with a failed run."""
        # Make model loading fail
        mock_model.side_effect = RuntimeError("Failed to load model")
        
        # Run the test
        result = test_local_model({
            "backend": "huggingface",
            "model_name_or_path": "invalid/model"
        })
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("Failed to load model", result["error"])


if __name__ == "__main__":
    unittest.main()

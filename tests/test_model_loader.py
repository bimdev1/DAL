"""Tests for the model_loader module."""

import unittest
from unittest.mock import patch, MagicMock
import torch
from dal.model_loader import (
    ModelBackend,
    ModelLoadError,
    load_local_model,
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
    def test_model_generation(self, mock_tokenizer, mock_model):
        """Test model generation with a mock model."""
        from dataclasses import dataclass
        from unittest.mock import MagicMock, ANY
        import torch
        
        # Create a mock GenerationResult
        @dataclass
        class MockGenerationResult:
            text: str
            tokens_prompt: int
            tokens_generated: int
            time_seconds: float
        
        # Create a mock model that will be returned by from_pretrained
        class MockModel:
            def __init__(self):
                self.model_info = {
                    'backend': 'huggingface',
                    'model_name': 'microsoft/phi-2',
                    'quantization': '4bit'
                }
                self.config = MagicMock()
                self.config.pad_token_id = 0
                self.config.eos_token_id = 1
                self.config.bos_token_id = 2
                
                # Create a mock for the generate method
                self.generate = MagicMock()
                
                # Create a mock tensor output for the model
                mock_output = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
                self.generate.return_value = mock_output
        
        # Create the mock model instance
        mock_model_instance = MockModel()
        mock_model.return_value = mock_model_instance
        
        # Create a mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer_instance.eos_token_id = 1
        mock_tokenizer_instance.bos_token_id = 2
        
        # Create a mock tensor with a to() method
        mock_tensor = MagicMock()
        mock_tensor.shape = [1, 3]  # Mock shape for input_ids
        mock_tensor.__getitem__.return_value = torch.tensor([1, 2, 3])  # For input_ids[1:]
        
        # Mock the return value of the tokenizer
        mock_encoded = MagicMock()
        mock_encoded.input_ids = mock_tensor
        mock_encoded.to.return_value = mock_encoded  # Make to() return self
        
        # Mock the tokenizer's return values
        mock_tokenizer_instance.return_value = mock_encoded
        mock_tokenizer_instance.decode.return_value = "Test output"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock the device
        with patch('torch.cuda.is_available', return_value=False):
            # Load the model
            model = load_local_model({
                "backend": "huggingface",
                "model_name_or_path": "microsoft/phi-2"
            })
            
            # Test generation with minimal parameters
            result = model.generate("Test prompt", max_new_tokens=10)
            
            # Verify the result
            self.assertEqual(result.text, "Test output")
            self.assertEqual(result.tokens_prompt, 3)  # Based on the mock input_ids
            self.assertEqual(result.tokens_generated, 12)  # 15 (output length) - 3 (input length)
        
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_model_loading_failure(self, mock_model):
        """Test model loading failure with an invalid model."""
        from dal.model_loader import ModelLoadError
        
        # Make model loading fail with a specific error message
        error_msg = "invalid/model is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'"
        mock_model.side_effect = RuntimeError(error_msg)
        
        # Test that loading an invalid model raises a ModelLoadError
        with self.assertRaises(ModelLoadError) as context:
            load_local_model({
                "backend": "huggingface",
                "model_name_or_path": "invalid/model"
            })
        
        # Verify the error message is wrapped in ModelLoadError
        self.assertIn(error_msg, str(context.exception))


if __name__ == "__main__":
    unittest.main()

"""Model loading and inference for local LLMs.

This module provides a unified interface for loading and running local language models
with support for multiple backends (Hugging Face, GGUF) and quantization.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

class ModelBackend(Enum):
    """Supported model backends."""
    HUGGINGFACE = "huggingface"
    GGUF = "gguf"

class ModelLoadError(Exception):
    """Raised when model loading fails."""
    def __init__(self, backend: ModelBackend, reason: str):
        self.backend = backend
        self.reason = reason
        super().__init__(f"Failed to load {backend.value} model: {reason}")

@dataclass
class GenerationResult:
    """Result of a model generation."""
    text: str
    tokens_generated: int
    tokens_prompt: int
    time_seconds: float

class LocalModel(ABC):
    """Abstract base class for local model implementations."""
    
    @property
    @abstractmethod
    def backend(self) -> ModelBackend:
        """Return the backend used by this model."""
        pass
    
    @property
    @abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """Return information about the loaded model."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate text from a prompt.
        
        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-1.0, higher=more random)
            stop_sequences: List of strings that will stop generation when encountered
            timeout_seconds: Maximum time to spend on generation
            **kwargs: Additional backend-specific arguments
            
        Returns:
            GenerationResult containing the generated text and metadata
            
        Raises:
            TimeoutError: If generation exceeds timeout_seconds
            RuntimeError: For other generation errors
        """
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.
        
        Default implementation uses a simple word-based approximation.
        Override in subclasses for accurate token counting.
        """
        # Very rough approximation: 1 token ≈ 4 chars or 1 word
        return max(len(text) // 4, len(text.split()))

class HuggingFaceModel(LocalModel):
    """Hugging Face model implementation."""
    
    def __init__(self, model_name_or_path: str, **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as e:
            raise ImportError(
                "Hugging Face dependencies not found. Install with: "
                "pip install torch transformers"
            ) from e
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Try to load with 4-bit quantization if available
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto"
            )
            self.quantization = "4bit"
        except (ImportError, RuntimeError):
            # Fall back to FP16 if quantization not available
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.quantization = "fp16"
        
        self._model_name = model_name_or_path.split("/")[-1]
        
    @property
    def backend(self) -> ModelBackend:
        return ModelBackend.HUGGINGFACE
    
    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "backend": self.backend.value,
            "model_name": self._model_name,
            "device": self.device,
            "quantization": self.quantization,
        }
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs
    ) -> GenerationResult:
        import torch
        from transformers import GenerationConfig
        
        start_time = time.time()
        
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_token_count = inputs.input_ids.shape[1]
        
        # Prepare generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            **kwargs
        )
        
        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    **kwargs
                )
            
            # Decode the output
            output_text = self.tokenizer.decode(
                outputs[0][input_token_count:], 
                skip_special_tokens=True
            )
            
            # Apply stop sequences if provided
            if stop_sequences:
                for stop in stop_sequences:
                    if stop in output_text:
                        output_text = output_text.split(stop)[0]
            
            output_token_count = outputs.shape[1] - input_token_count
            
            return GenerationResult(
                text=output_text,
                tokens_generated=output_token_count,
                tokens_prompt=input_token_count,
                time_seconds=time.time() - start_time
            )
            
        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}") from e

class GGUFModel(LocalModel):
    """GGUF model implementation using llama-cpp-python."""
    
    def __init__(self, model_path: str, **kwargs):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python not found. Install with: "
                "pip install llama-cpp-python"
            ) from e
        
        # Default params
        self.n_ctx = kwargs.pop("n_ctx", 2048)
        self.n_threads = kwargs.pop("n_threads", 4)
        self.n_gpu_layers = kwargs.pop("n_gpu_layers", 0)  # 0 = CPU only
        
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                **kwargs
            )
        except Exception as e:
            raise ModelLoadError(
                ModelBackend.GGUF,
                f"Failed to load GGUF model: {str(e)}"
            )
        
        self._model_path = model_path
        self._model_name = model_path.split("/")[-1]
    
    @property
    def backend(self) -> ModelBackend:
        return ModelBackend.GGUF
    
    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "backend": self.backend.value,
            "model_path": self._model_path,
            "model_name": self._model_name,
            "context_length": self.n_ctx,
            "threads": self.n_threads,
            "gpu_layers": self.n_gpu_layers,
        }
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs
    ) -> GenerationResult:
        start_time = time.time()
        
        try:
            # Count input tokens
            input_token_count = len(self.llm.tokenize(prompt.encode("utf-8")))
            
            # Prepare kwargs for generation
            gen_kwargs = {
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            }
            
            # Only add stop_sequences if not None and not empty
            if stop_sequences:
                gen_kwargs["stop"] = stop_sequences
                
            # Add any additional kwargs
            gen_kwargs.update(kwargs)
            
            # Generate the output
            output = self.llm(**gen_kwargs)
            
            if isinstance(output, dict) and "choices" in output:
                output_text = output["choices"][0]["text"]
                output_token_count = len(self.llm.tokenize(output_text.encode("utf-8")))
            else:
                output_text = str(output)
                output_token_count = 0
            
            return GenerationResult(
                text=output_text,
                tokens_generated=output_token_count,
                tokens_prompt=input_token_count,
                time_seconds=time.time() - start_time
            )
            
        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}") from e

def load_local_model(spec: dict) -> LocalModel:
    """Load a local model from a specification dictionary.
    
    Args:
        spec: Dictionary containing model loading parameters:
            - backend: "huggingface" or "gguf"
            - model_name_or_path: Model identifier (HF model ID or path to GGUF file)
            - device: "cuda" or "cpu" (for HF)
            - n_threads: Number of CPU threads (for GGUF)
            - n_ctx: Context length (for GGUF)
            - n_gpu_layers: Number of layers to offload to GPU (for GGUF)
            
    Returns:
        A LocalModel instance for the specified backend
        
    Raises:
        ValueError: For invalid backend or missing required parameters
        ModelLoadError: If model loading fails
        ImportError: If required dependencies are missing
    """
    backend = spec.get("backend", "huggingface").lower()
    model_name_or_path = spec.get("model_name_or_path")
    
    if not model_name_or_path:
        raise ValueError("model_name_or_path is required")
    
    try:
        if backend == "huggingface":
            # Remove model_name_or_path from spec to avoid duplicate argument
            spec_copy = spec.copy()
            spec_copy.pop('model_name_or_path', None)
            return HuggingFaceModel(model_name_or_path, **spec_copy)
        elif backend == "gguf":
            return GGUFModel(model_name_or_path, **spec)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    except Exception as e:
        if isinstance(e, (ValueError, ImportError)):
            raise
        raise ModelLoadError(
            ModelBackend(backend),
            str(e)
        ) from e

# Note: The test_local_model function has been moved to tests/test_model_loader.py

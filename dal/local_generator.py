"""Local Generator for DAL.

This module provides a high-level interface for generating text using local models
with SDT (Segment Descriptor Tag) based prompting.
"""

import time
import logging
from typing import Dict, Optional, Any, List
from dataclasses import asdict

from .model_loader import load_local_model, LocalModel
from .sdt_injector import build_prompt, SDTApplied
from .dal_types import DALTagBlock

logger = logging.getLogger(__name__)


class LocalGenerator:
    """Handles local text generation with SDT-based prompting."""

    def __init__(self, model_spec: Dict[str, Any]) -> None:
        """Initialize the local generator with a model specification.
        
        Args:
            model_spec: Dictionary containing model configuration
        """
        self.model_spec = model_spec
        self.model: Optional[LocalModel] = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the local model."""
        try:
            self.model = load_local_model(self.model_spec)
            logger.info("Loaded local model: %s", self.model_spec.get('model_name_or_path'))
        except Exception as e:
            logger.error("Failed to load local model: %s", str(e))
            self.model = None
    
    def is_ready(self) -> bool:
        """Check if the generator is ready to generate text."""
        return self.model is not None
    
    def generate(
        self,
        text: str,
        sdt: DALTagBlock,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        timeout_s: float = 8.0,
        sdt_opts: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Generate text based on input text and SDT.
        
        Args:
            text: Input text to process
            sdt: DALTagBlock containing SDT attributes
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            stop: List of stop sequences
            timeout_s: Maximum time to wait for generation in seconds
            sdt_opts: Optional SDT options for prompt construction
            
        Returns:
            Dictionary containing:
            - text: Generated text (or original text on failure)
            - tokens_in: Number of input tokens
            - tokens_out: Number of output tokens
            - duration: Generation time in seconds
            - backend: Backend used (hf/gguf)
            - model: Model name/identifier
            - prompt_preview: Truncated prompt for logging
            - sdt_applied: Normalized SDT attributes used (dict)
        """
        if not self.is_ready():
            logger.warning("Local model not loaded, falling back to passthrough")
            return self._passthrough_result(text)
        
        # Build the prompt using SDT
        prompt, sdt_applied = build_prompt(text, sdt, sdt_opts)
        prompt_preview = (prompt[:200] + "...") if len(prompt) > 200 else prompt
        
        # Prepare generation parameters
        gen_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "stop": stop or [],
            "timeout_s": timeout_s
        }
        
        try:
            start_time = time.time()
            
            # Generate text
            result = self.model.generate(
                prompt=prompt,
                max_new_tokens=gen_params["max_new_tokens"],
                temperature=gen_params["temperature"],
                stop=gen_params["stop"]
            )
            
            duration = time.time() - start_time
            
            # Convert SDTApplied to dict for serialization
            sdt_applied_dict = sdt_applied.to_dict() if sdt_applied else None
            
            return {
                "text": result.text,  # Access attributes directly from the GenerationResult object
                "tokens_in": result.tokens_prompt,
                "tokens_out": result.tokens_generated,
                "duration": duration,
                "backend": self.model_spec.get("backend", "unknown"),
                "model": self.model_spec.get("model_name_or_path", "unknown"),
                "prompt_preview": prompt_preview,
                "sdt_applied": sdt_applied_dict
            }
            
        except Exception as e:
            logger.error("Generation failed: %s", str(e), exc_info=True)
            return self._passthrough_result(text, error=str(e))
    
    def _passthrough_result(
        self, 
        text: str, 
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return a passthrough result with error information.
        
        Args:
            text: The original text to pass through
            error: Optional error message
            
        Returns:
            Dictionary with passthrough result
        """
        if error:
            logger.warning("Falling back to passthrough: %s", error)
            
        return {
            "text": text,
            "tokens_in": 0,
            "tokens_out": 0,
            "duration": 0.0,
            "backend": "passthrough",
            "model": "none",
            "prompt_preview": "",
            "sdt_applied": None,
            "error": error or "Model not loaded"
        }

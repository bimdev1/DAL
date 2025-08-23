"""Improved expansion layer for DAL segments with dynamic processing.

This module enhances the base Expander with dynamic prompt segmentation and
adaptive generation parameters based on input length. It implements the
improvements from DAL Patch 2, including:

- Dynamic segmentation based on input length
- Adaptive max_length calculation
- Better handling of short and long inputs
- Improved error handling and fallbacks
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any, Tuple
import time

logger = logging.getLogger(__name__)


def split_prompt(prompt: str, max_segments: int) -> List[str]:
    """
    Split the input prompt into up to max_segments segments of roughly
    equal word length. Splitting occurs at word boundaries to avoid cutting
    phrases in half. If fewer words are present than segments, the original
    prompt is returned as a single segment.
    """
    words = prompt.split()
    if max_segments <= 1 or len(words) <= 1:
        return [prompt]

    segment_length = max(1, len(words) // max_segments)
    segments = []
    for i in range(0, len(words), segment_length):
        segments.append(" ".join(words[i : i + segment_length]))
        if len(segments) == max_segments - 1:
            remainder = " ".join(words[i + segment_length :])
            if remainder:
                segments[-1] = " ".join([segments[-1], remainder])
            break
    return segments


def determine_segments(token_count: int) -> int:
    """
    Decide how many segments to use based on the token count of the input.
    Shorter inputs remain in a single segment, while longer inputs are
    divided to aid the generation model.
    """
    if token_count < 512:
        return 1
    if token_count < 1024:
        return 2
    return 3


class ExpanderV2:
    """Enhanced Expander with dynamic segmentation and adaptive processing."""

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6") -> None:
        """Initialize the enhanced expander with dynamic processing."""
        self.model_name = model_name
        self._summariser = None
        self._tokenizer = None
        self._device = 0  # Default to GPU if available

    def _load_model(self) -> None:
        """Lazily load the model and tokenizer."""
        if self._summariser is not None and self._tokenizer is not None:
            return

        try:
            from transformers import pipeline, AutoTokenizer
            import torch
            
            # Check for CUDA availability
            if not torch.cuda.is_available():
                self._device = -1  # Use CPU
                logger.warning("CUDA not available, falling back to CPU")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._summariser = pipeline(
                "summarization",
                model=self.model_name,
                tokenizer=self._tokenizer,
                device=self._device,
                framework="pt"
            )
            logger.info("Loaded summarization model %s (device: %s)", 
                      self.model_name, "GPU" if self._device >= 0 else "CPU")
            
        except Exception as exc:
            logger.error(
                "Could not load summarization model '%s': %s", self.model_name, exc,
                exc_info=logger.isEnabledFor(logging.DEBUG)
            )
            self._summariser = None
            self._tokenizer = None

    def _calculate_length_parameters(self, input_tokens: int) -> Tuple[int, int]:
        """
        Calculate target and minimum lengths for generation based on input tokens.
        Ensures max_length > input_tokens only when we want to expand.
        
        Args:
            input_tokens: Number of tokens in the input
            
        Returns:
            Tuple of (target_length, min_length)
        """
        # For very short inputs (questions, commands) - don't expand
        if input_tokens < 15:
            # For these, we'll skip expansion entirely in _process_segment
            return input_tokens, max(4, input_tokens // 2)
            
        # For short but complete thoughts - slight expansion or maintain
        elif input_tokens < 30:
            target = min(64, max(input_tokens + 5, int(input_tokens * 1.2)))
            
        # For medium inputs - moderate expansion
        elif input_tokens < 100:
            target = min(128, int(input_tokens * 1.1))
            
        # For long inputs - compress
        else:
            target = min(256, int(input_tokens * 0.7))
        
        # Ensure min_length makes sense (at least 8, at most 64, and not more than target)
        min_length = max(8, min(min(64, target - 4), input_tokens // 2))
        
        # Ensure target is at least min_length + 1
        target = max(target, min_length + 1)
        
        return target, min_length

    def _process_segment(self, segment: str, max_length_hint: Optional[int] = None) -> str:
        """
        Process a single segment with the loaded model.
        
        Args:
            segment: The text segment to process
            max_length_hint: Optional maximum length suggestion from caller
            
        Returns:
            Processed text
        """
        if not self._summariser or not self._tokenizer:
            return segment
            
        try:
            # Calculate input tokens
            inputs = self._tokenizer(segment, return_tensors="pt")
            input_tokens = inputs.input_ids.size(1)
            
            # Get target lengths
            target_length, min_length = self._calculate_length_parameters(input_tokens)
            
            # Skip processing if:
            # 1. Very short segment that won't benefit from expansion
            # 2. Target length isn't meaningfully different from input
            if (input_tokens < 15) or (abs(target_length - input_tokens) < 5):
                logger.debug("Skipping expansion for short/similar length segment (%d → %d tokens)", 
                           input_tokens, target_length)
                return segment
            
            # Respect caller's max_length_hint if provided
            if max_length_hint is not None:
                target_length = min(target_length, max_length_hint)
                
            logger.debug(
                "Processing segment: %d tokens -> target_length=%d, min_length=%d",
                input_tokens, target_length, min_length
            )
            
            # Generate with tuned parameters
            result = self._summariser(
                segment,
                max_length=target_length,
                min_length=min_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                length_penalty=1.0,
                num_beams=4,
                early_stopping=True,
                truncation=True
            )
            
            # Clean up the output
            output = result[0]['summary_text'].strip()
            
            # If output is empty or just repeats the input, return original
            if not output or output.lower() == segment.lower():
                return segment
                
            return output
            
        except Exception as exc:
            logger.warning(
                "Error during segment expansion: %s", exc,
                exc_info=logger.isEnabledFor(logging.DEBUG)
            )
            return segment

    def expand(self, segment: str, tags: Optional[List[str]] = None) -> str:
        """
        Expand a segment using dynamic processing.
        
        Args:
            segment: The text to expand
            tags: Optional list of tags for the segment (not currently used)
            
        Returns:
            Expanded version of the input text
        """
        segment = segment.strip()
        if not segment:
            return segment
            
        self._load_model()
        if not self._summariser or not self._tokenizer:
            return segment
            
        # Tokenize to determine processing strategy
        try:
            tokens = self._tokenizer(segment, return_tensors="pt").input_ids
            token_count = tokens.size(1)
            
            # Skip processing for very short inputs
            if token_count < 12:  # ~8-10 words
                logger.debug("Skipping expansion for very short input (%d tokens)", token_count)
                return segment
                
            # Determine processing parameters
            num_segments = determine_segments(token_count)
            max_length_hint = min(256, int(token_count * 1.5))  # Sane upper bound
            
            # Split and process segments
            segments = split_prompt(segment, num_segments)
            expanded_segments = [
                self._process_segment(s, max_length_hint) 
                for s in segments
            ]
            
            # Combine results with proper spacing
            return " ".join(s for s in expanded_segments if s.strip())
            
        except Exception as exc:
            logger.warning(
                "Error in expand(): %s", exc,
                exc_info=logger.isEnabledFor(logging.DEBUG)
            )
            return segment

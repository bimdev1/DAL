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
from typing import List, Optional, Dict, Any

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


def determine_max_length(token_count: int, upper_bound: int = 1024) -> int:
    """
    Compute a sensible max_length parameter for sequence generation
    relative to the number of input tokens.
    """
    proposed = int(token_count * 1.5)
    return max(128, min(proposed, upper_bound))


class ExpanderV2:
    """Enhanced Expander with dynamic segmentation and adaptive processing."""

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6") -> None:
        """Initialize the enhanced expander with dynamic processing."""
        self.model_name = model_name
        self._summariser = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazily load the model and tokenizer."""
        if self._summariser is not None and self._tokenizer is not None:
            return

        try:
            from transformers import pipeline, AutoTokenizer
            
            self._summariser = pipeline(
                "summarization",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0,  # Use GPU if available
                framework="pt"
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("Loaded summarization model %s", self.model_name)
            
        except Exception as exc:
            logger.warning(
                "Could not load summarization model '%s': %s", self.model_name, exc
            )
            self._summariser = None
            self._tokenizer = None

    def _process_segment(self, segment: str, max_length: int) -> str:
        """Process a single segment with the loaded model."""
        if not self._summariser:
            return segment
            
        try:
            result = self._summariser(
                segment,
                max_length=max_length,
                min_length=max(32, max_length // 2),
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                truncation=True
            )
            return result[0]['summary_text']
        except Exception as exc:
            logger.warning("Error during segment expansion: %s", exc)
            return segment

    def expand(self, segment: str, tags: Optional[List[str]] = None) -> str:
        """Expand a segment using dynamic processing.
        
        Args:
            segment: The text to expand
            tags: Optional list of tags for the segment
            
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
        tokens = self._tokenizer(segment, return_tensors="pt").input_ids
        token_count = tokens.size(1)
        
        # Determine processing parameters
        num_segments = determine_segments(token_count)
        max_length = determine_max_length(token_count // num_segments)
        
        # Split and process segments
        segments = split_prompt(segment, num_segments)
        expanded_segments = [self._process_segment(s, max_length) for s in segments]
        
        # Combine results
        return " ".join(expanded_segments)

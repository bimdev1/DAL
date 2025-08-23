"""Optional expansion layer for DAL segments.

This module defines the :class:`Expander` class, which takes DAL
segments and attempts to expand them into richer, more detailed
explanations.  In a full DAL system this would be accomplished by
calling a small (1–3 billion parameter) language model such as
Phi‑2 or TinyLLaMA.  Since this environment may not include
pre‑installed models, the implementation here tries to load a
pretrained summarisation pipeline from Hugging Face if available.  If
the required dependencies are not present, the expander falls back
to returning the original segment unchanged.

Example usage::

    from dal.expander import Expander

    expander = Expander()
    expanded = expander.expand(
        "Economic factors can contribute to system failures.",
        tags=["economy"]
    )
    print(expanded)

The expander will attempt to generate a more detailed sentence
describing how economic factors lead to failures.  If summarisation
models are unavailable, it simply returns the input segment.
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class Expander:
    """Expand DAL segments using a small language model.

    The expander loads a summarisation pipeline from Hugging Face's
    ``transformers`` library the first time it is used.  Loading the
    model can be slow; therefore the pipeline is cached.  If the
    ``transformers`` package is not installed or loading fails, the
    expander will operate in a no‑op mode and return the original
    segment text.
    """

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6") -> None:
        """Initialise the expander.

        Args:
            model_name: The Hugging Face model identifier for the
                summarisation pipeline.  Defaults to a small BART
                variant.  You can replace this with the path to a
                local 1–3 billion parameter model once available.
        """
        self.model_name = model_name
        self._summariser = None

    def _load_summariser(self) -> None:
        """Lazily load the summarisation pipeline.

        If ``transformers`` is not installed or the model cannot be
        loaded, the internal summariser remains ``None`` and the
        expander will fall back to no‑op expansion.
        """
        if self._summariser is not None:
            return
        try:
            from transformers import pipeline  # type: ignore
            self._summariser = pipeline(
                "summarization",
                model=self.model_name,
                tokenizer=self.model_name,
            )
            logger.info("Loaded summarisation model %s", self.model_name)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Could not load summarisation model '%s': %s", self.model_name, exc
            )
            self._summariser = None

    def expand(self, segment: str, tags: Optional[List[str]] | None = None) -> str:
        """Expand a single segment into a richer explanation.

        Args:
            segment: The segment text to expand.
            tags: The tags associated with the segment.  These can be
                used in future implementations to guide the expansion.

        Returns:
            A potentially longer, more detailed version of the segment.
        """
        # Trim whitespace
        segment = segment.strip()
        if not segment:
            return segment
        self._load_summariser()
        if self._summariser is None:
            # Fallback: return original segment unchanged
            return segment
        try:
            # Use the model to "summarise" the segment.  Summarisation
            # here actually acts as an expander because we can request
            # a longer max length than the input.  We set min_length
            # to be slightly larger than the input length to encourage
            # expansion.
            # Compute reasonable lengths based on token count
            min_len = max(20, len(segment.split()) + 5)
            max_len = min_len + 30
            summary_list = self._summariser(
                segment,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
            )
            if summary_list and isinstance(summary_list[0], dict):
                expanded = summary_list[0].get("summary_text", segment)
            else:
                expanded = segment
            return expanded.strip()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Summarisation failed: %s", exc)
            return segment
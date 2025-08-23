"""High‑level DAL pipeline orchestrator.

This module ties together the individual components of the DAL
implementation to process a natural language prompt from start to
finish.  It segments the prompt, assigns tags, builds a tag
transition model and stitches the final answer.  The API is kept
minimal to support experimentation and composability.

Functions
---------

run_pipeline(prompt: str, n_segments: int = 3) -> dict
    Process a prompt through the DAL pipeline and return intermediate
    and final results.
"""

from __future__ import annotations

from typing import Dict, List

from .segmenter import segment_prompt
from .tagging import assign_tags
from .stitcher import stitch_segments
from .vectorizer import simple_vectorise


def run_pipeline(
    prompt: str,
    n_segments: int = 3,
    *,
    expand: bool = False,
    show_vectors: bool = False,
) -> Dict[str, object]:
    """Run a prompt through the simplified DAL pipeline.

    The pipeline performs the following steps:

    1. Splits the prompt into ``n_segments`` segments based on sentence
       boundaries.
    2. Assigns keyword‑based tags to each segment.
    3. Computes a simple hashed vector for each segment (for future
       use; optionally returned when ``show_vectors`` is True).
    4. Optionally expands each segment using a small local LLM via the
       ``Expander`` module.  If expansion is enabled, the expanded
       segments replace the originals for stitching.
    5. Orders and stitches the segments into a final answer using a
       simple Markov model over tags.

    Args:
        prompt: The natural language prompt to process.
        n_segments: The number of segments into which the prompt
            should be divided.
        expand: If ``True``, use the ``Expander`` to expand each segment
            prior to stitching.  Requires that the optional
            ``transformers`` dependency is installed and that a small
            summarisation model is available.  When ``False``, no
            expansion is performed.
        show_vectors: If ``True``, include the computed vectors in the
            returned dictionary.

    Returns:
        A dictionary containing the original segments, the assigned tags
        for each segment, the computed vectors (if requested), any
        expanded segments and the final stitched answer.
    """
    # Step 1: segmentation
    segments: List[str] = segment_prompt(prompt, n_segments)
    # Step 2: tagging
    tags: List[List[str]] = [assign_tags(seg) for seg in segments]
    # Step 3: vectorisation
    vectors: List[List[float]] | None = None
    if show_vectors:
        vectors = [simple_vectorise(seg) for seg in segments]
    # Step 4: optional expansion
    expanded_segments: List[str] | None = None
    if expand:
        try:
            # Import here to avoid dependency overhead when not used
            from .expander import Expander

            expander = Expander()
            expanded_segments = []
            for seg in segments:
                try:
                    expanded = expander.expand(seg)
                except Exception:
                    # If expansion fails, fall back to original
                    expanded = seg
                expanded_segments.append(expanded)
        except Exception:
            # If Expander cannot be loaded (missing transformers), skip
            expanded_segments = None
    # Determine which segments to use for stitching
    segments_for_stitch = expanded_segments if expanded_segments is not None else segments
    # Step 5: stitching
    final_answer: str = stitch_segments(segments_for_stitch, tags)
    result: Dict[str, object] = {
        "segments": segments,
        "tags": tags,
        "answer": final_answer,
    }
    if show_vectors:
        result["vectors"] = vectors if vectors is not None else []
    if expanded_segments is not None:
        result["expanded_segments"] = expanded_segments
    return result
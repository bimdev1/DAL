"""Prompt segmentation for DAL.

The DAL pipeline operates on discrete segments of the input.  This
module provides a simple utility for splitting a natural language
prompt into a specified number of segments.  It uses sentence
boundaries to form segments and attempts to balance them in length.

Functions
---------

segment_prompt(text: str, n_segments: int) -> list[str]
    Split the input text into ``n_segments`` overlapping chunks.
"""

from __future__ import annotations

import re
from typing import List


_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


def segment_prompt(text: str, n_segments: int) -> List[str]:
    """Split a prompt into a number of segments.

    The function first breaks the input on sentence boundaries.  It
    then groups sentences into ``n_segments`` approximately equal
    parts.  If ``n_segments`` is greater than the number of
    sentences, empty segments will be padded.  Segment boundaries are
    not strict; neighbouring segments may share a boundary sentence
    when the division is not exact.

    Args:
        text: The prompt to segment.
        n_segments: The desired number of segments.

    Returns:
        A list of ``n_segments`` strings.
    """
    sentences = [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]
    if not sentences:
        return [text] + [""] * (n_segments - 1)
    # Determine approximate segment size
    seg_size = max(1, len(sentences) // n_segments)
    segments: List[str] = []
    i = 0
    for seg_idx in range(n_segments):
        # For the last segment take all remaining sentences
        if seg_idx == n_segments - 1:
            chunk = sentences[i:]
        else:
            chunk = sentences[i:i + seg_size]
        segments.append(" ".join(chunk))
        i += seg_size
    # Pad with empty strings if needed
    while len(segments) < n_segments:
        segments.append("")
    return segments
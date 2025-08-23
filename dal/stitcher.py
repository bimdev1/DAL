"""Assemble DAL segments into a final answer.

The stitching logic combines segments produced during the DAL
pipeline.  It uses the sequence of assigned tags to build a simple
Markov model of tag transitions and then orders the segments by
descending tag frequency.  It also removes duplicate sentences and
overlapping text to produce a clean answer.

Functions
---------

stitch_segments(segments: list[str], tags: list[list[str]]) -> str
    Assemble segments into a final answer string.
"""

from __future__ import annotations

import itertools
from typing import List

from .markov import MarkovChain


def _flatten_tags(tags: List[List[str]]) -> List[str]:
    """Flatten a list of tag lists into a single list by choosing
    the first tag from each segment.  If a segment has no tags, a
    placeholder tag ``"misc"`` is used.  The resulting list of tags
    has the same length as the number of segments.
    """
    result: List[str] = []
    for tag_list in tags:
        if tag_list:
            result.append(tag_list[0])
        else:
            result.append("misc")
    return result


def stitch_segments(segments: List[str], tags: List[List[str]]) -> str:
    """Combine segment texts into a single stitched answer.

    This function builds a Markov chain from the sequence of tags and
    uses tag frequencies to determine a reasonable ordering of
    segments.  The segments with more common tags will appear
    earlier.  Duplicate sentences across segments are deduplicated.

    Args:
        segments: A list of segment strings.
        tags: A parallel list of tag lists for each segment.  Only the
            first tag from each list is considered for ordering.

    Returns:
        A single string representing the combined answer.
    """
    if not segments:
        return ""
    # Flatten tags and count frequencies
    flat_tags = _flatten_tags(tags)
    # Build a Markov chain on tags for demonstration
    mc = MarkovChain(set(flat_tags))
    mc.observe_sequence(flat_tags)
    mc.normalize()
    # Determine ordering by sorting segments by tag frequency (descending)
    tag_counts = {tag: flat_tags.count(tag) for tag in set(flat_tags)}
    ordered_indices = sorted(range(len(segments)), key=lambda i: (-tag_counts[flat_tags[i]], i))
    ordered_segments = [segments[i] for i in ordered_indices]
    # Remove duplicate sentences while preserving order
    seen_sentences = set()
    final_sentences: List[str] = []
    for seg in ordered_segments:
        for sentence in filter(None, (s.strip() for s in seg.split('.'))):
            if sentence and sentence not in seen_sentences:
                final_sentences.append(sentence)
                seen_sentences.add(sentence)
    return '. '.join(final_sentences) + ('.' if final_sentences else '')
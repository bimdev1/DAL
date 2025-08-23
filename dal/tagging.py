"""Simple keyword‑based tagging for DAL segments.

This module defines a minimal tagging system that assigns high‑level
labels to segments of text based on the presence of certain
keywords.  In a production DAL implementation, this logic would be
replaced by a learned classifier or a more comprehensive rule base.
The tags help the Markov chain and stitching logic understand the
semantic content of each segment.

Functions
---------

assign_tags(text: str) -> list[str]
    Analyse the input text and return a list of assigned tags.
"""

from __future__ import annotations

import re
from typing import List


# Map of tags to the set of keywords that signal that tag.
# This dictionary is deliberately small; extend it to suit your
# domain.  Keywords should be lowercase.
KEYWORD_TAGS = {
    "economy": {"economy", "economic", "money", "finance", "market"},
    "politics": {"politics", "policy", "government", "state", "election"},
    "conspiracy": {"conspiracy", "secret", "hidden", "cabal", "cover‑up"},
    "collapse": {"collapse", "breakdown", "failure", "crash", "decline"},
    "technology": {"technology", "tech", "ai", "software", "innovation"},
    "society": {"society", "social", "culture", "community", "people"},
    "environment": {"environment", "climate", "ecology", "green", "pollution"},
}

# Regular expression to extract words for matching
_WORD_RE = re.compile(r"\b\w+\b")


def assign_tags(text: str) -> List[str]:
    """Assign high‑level tags based on keyword presence.

    The function tokenises the input into lower‑case words and checks
    each word against the keyword sets defined in ``KEYWORD_TAGS``.
    Tags whose keywords appear at least once are included in the
    returned list.  If no keywords match, a single ``"misc"`` tag
    is returned.

    Args:
        text: The text to analyse.

    Returns:
        A list of tag strings.  Ordering is undefined.
    """
    words = set(w.lower() for w in _WORD_RE.findall(text))
    tags: List[str] = []
    for tag, keywords in KEYWORD_TAGS.items():
        if any(word in words for word in keywords):
            tags.append(tag)
    if not tags:
        tags.append("misc")
    return tags
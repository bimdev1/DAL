"""Simple text vectorisation utilities.

The full DAL vision envisions converting a natural language prompt
into a dense, machine‑friendly vector that captures its semantics.
Here we provide a minimal implementation that maps each token in a
string to a hashed vector representation.  The resulting vector is
deterministic for a given input but does not rely on any external
models.  It serves as a placeholder for future integration with
pre‑trained embeddings.

Functions
---------

simple_vectorise(text: str, dim: int = 16) -> list[float]
    Convert the input text into a fixed‑length numeric vector using
    token hashing.
"""

from __future__ import annotations

import hashlib
import re
from typing import List


_WORD_RE = re.compile(r"\b\w+\b")


def _hash_word(word: str, dim: int) -> List[int]:
    """Hash a single word into a vector of integer buckets.

    We use MD5 to compute a digest of the word.  Each byte of the
    digest contributes to a bucket in the output vector by taking the
    modulo of the byte value with ``dim``.  This yields a sparse
    representation where the counts correspond loosely to token
    presence.

    Args:
        word: The word to hash.
        dim: The dimension of the output vector.

    Returns:
        A list of length ``dim`` containing integer counts.
    """
    digest = hashlib.md5(word.encode("utf-8")).digest()
    buckets = [0] * dim
    # Distribute each byte into a bucket
    for b in digest:
        buckets[b % dim] += 1
    return buckets


def simple_vectorise(text: str, dim: int = 16) -> List[float]:
    """Vectorise text into a fixed‑length numeric representation.

    The function tokenises the input on word boundaries, hashes each
    word to a small vector and sums these vectors.  The resulting
    vector is then normalised by the total number of tokens so that
    it can be compared across different inputs.  Note that this
    scheme does not capture semantics; it is intended as a stand‑in
    until a more sophisticated embedding can be integrated.

    Args:
        text: The input string to vectorise.
        dim: The dimensionality of the output vector.

    Returns:
        A list of length ``dim`` containing floats between 0 and 1.
    """
    words = _WORD_RE.findall(text.lower())
    if not words:
        return [0.0] * dim
    accum = [0] * dim
    for word in words:
        hashed = _hash_word(word, dim)
        for i, val in enumerate(hashed):
            accum[i] += val
    # Normalise by total count
    total = sum(accum)
    return [val / total for val in accum]
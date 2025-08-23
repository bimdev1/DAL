"""
Improved prompt processing for the WindSurf cascade pipeline.

This module introduces dynamic prompt segmentation and adaptive `max_length`
selection to improve output quality and performance. It implements the
recommendations derived from benchmarking the DistilBART cascade, such as
reducing unnecessary segments for short inputs and choosing a sensible
generation length based on the size of the input.

To integrate these improvements into an existing pipeline, import
`process_prompt` and call it in place of the previous generation logic. The
helper functions `determine_segments`, `determine_max_length` and
`split_prompt` are provided for fine‑tuning or reuse in other contexts.
"""

from typing import List, Optional


def split_prompt(prompt: str, max_segments: int) -> List[str]:
    """
    Split the input prompt into up to ``max_segments`` segments of roughly
    equal word length. Splitting occurs at word boundaries to avoid cutting
    phrases in half. If fewer words are present than segments, the original
    prompt is returned as a single segment.

    Args:
        prompt: The raw text to segment.
        max_segments: The maximum number of segments desired.

    Returns:
        A list of string segments.
    """
    words = prompt.split()
    # If segmentation is unnecessary, return the whole prompt.
    if max_segments <= 1 or len(words) <= 1:
        return [prompt]

    # Compute an approximate length for each segment.
    segment_length = max(1, len(words) // max_segments)
    segments = []
    for i in range(0, len(words), segment_length):
        # Compose the current segment from the slice of words.
        segments.append(" ".join(words[i : i + segment_length]))
        # If we've reached the desired number of segments minus one, append
        # the remainder of the prompt into the final segment and break.
        if len(segments) == max_segments - 1:
            remainder = " ".join(words[i + segment_length :])
            if remainder:
                segments[-1] = " ".join([segments[-1], remainder])
            break
    return segments


def determine_segments(token_count: int) -> int:
    """
    Decide how many segments to use based on the encoded token count of
    the input. Short inputs remain in a single segment, while longer inputs
    are divided to aid the generation model in focusing on smaller chunks.

    Args:
        token_count: The number of tokens in the encoded prompt.

    Returns:
        An integer representing the suggested number of segments.
    """
    if token_count < 512:
        return 1
    if token_count < 1024:
        return 2
    return 3


def determine_max_length(token_count: int, upper_bound: int = 1024) -> int:
    """
    Compute a sensible ``max_length`` parameter for sequence generation
    relative to the number of input tokens. The default heuristic returns
    1.5× the input length, bounded between 128 and ``upper_bound``. This
    prevents unnecessarily long generation calls for short prompts and
    avoids excessive memory use for very long inputs.

    Args:
        token_count: The number of tokens in the input sequence.
        upper_bound: The maximum allowed ``max_length`` value.

    Returns:
        An integer suitable for the ``max_length`` argument to
        ``model.generate``.
    """
    proposed = int(token_count * 1.5)
    # Ensure a reasonable lower and upper bound.
    return max(128, min(proposed, upper_bound))


def process_prompt(
    prompt: str,
    model,
    tokenizer,
    segments: Optional[int] = None,
    max_length: Optional[int] = None,
) -> str:
    """
    Generate a response for the provided ``prompt`` using a segmentation
    strategy and dynamic ``max_length`` computation. This function is
    intended as a drop‑in replacement for generic generation logic in the
    WindSurf cascade.

    The input prompt is first encoded to determine its length in tokens.
    Unless explicitly overridden, the number of segments and the generation
    length are chosen automatically based on the input length. The prompt
    is then segmented on word boundaries, and each segment is processed
    separately through the model. The individual outputs are concatenated
    with blank lines in between to preserve separation.

    Args:
        prompt: The text to be expanded by the language model.
        model: A HuggingFace model instance with a ``generate`` method.
        tokenizer: The corresponding tokenizer with an ``encode`` method.
        segments: Optional override for the number of segments to divide the
            prompt into. If ``None``, the value from ``determine_segments``
            will be used.
        max_length: Optional override for the ``max_length`` argument used
            during generation. If ``None``, the value from
            ``determine_max_length`` will be used.

    Returns:
        The concatenated generated output string.
    """
    # Encode the entire prompt once to determine token count.
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    token_count = input_ids.size(-1)

    # Auto‑select number of segments if not specified.
    if segments is None:
        segments = determine_segments(token_count)

    # Auto‑select max_length if not specified.
    if max_length is None:
        max_length = determine_max_length(token_count)

    # Split the prompt into the requested number of segments.
    prompt_segments = split_prompt(prompt, segments)

    outputs: List[str] = []
    for segment in prompt_segments:
        segment_ids = tokenizer.encode(segment, return_tensors="pt")
        generated = model.generate(
            segment_ids,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        outputs.append(text.strip())

    return "\n\n".join(outputs)
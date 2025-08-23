#!/usr/bin/env python3
"""Command line interface for the simplified DAL system.

This script provides a convenient way to run the DAL pipeline from
the command line.  It accepts a prompt via argument or standard
input and prints out the intermediate segments, tags and the final
answer.  Use this script to experiment with different prompts and
numbers of segments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dal.pipeline import run_pipeline


def _read_prompt(args: argparse.Namespace) -> str:
    """Read the prompt from the CLI argument or stdin.

    If the ``--prompt`` argument is provided, its value is used.
    Otherwise the script reads from standard input until EOF.
    """
    if args.prompt is not None:
        return args.prompt
    # Read all lines from stdin
    return sys.stdin.read().strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a prompt through the DAL pipeline")
    parser.add_argument(
        "--prompt",
        type=str,
        help="The prompt to process.  If omitted, read from standard input.",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=3,
        help="Number of segments to split the prompt into (default: 3)",
    )
    parser.add_argument(
        "--show-vectors",
        dest="show_vectors",
        action="store_true",
        help="Print the numeric vectors for each segment",
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help=(
            "Expand each segment using a small local LLM before stitching; "
            "requires optional dependencies."
        ),
    )
    args = parser.parse_args()
    prompt = _read_prompt(args)
    if not prompt:
        print("No prompt provided", file=sys.stderr)
        sys.exit(1)
    result = run_pipeline(
        prompt,
        n_segments=args.segments,
        expand=args.expand,
        show_vectors=args.show_vectors,
    )
    segments = result["segments"]
    tags = result["tags"]
    answer = result["answer"]
    for idx, (seg, tag_list) in enumerate(zip(segments, tags), start=1):
        print(f"\nSegment {idx} (tags: {', '.join(tag_list)}):\n{seg}")
        if args.show_vectors:
            vectors = result.get("vectors", [])
            if vectors:
                print(f"Vector {idx}: {vectors[idx-1]}")
        # If expanded segments were generated, show them for debugging
        expanded_segments = result.get("expanded_segments")
        if expanded_segments is not None:
            print(f"Expanded {idx}: {expanded_segments[idx-1]}")
    print("\nStitched answer:\n" + answer)


if __name__ == "__main__":
    main()
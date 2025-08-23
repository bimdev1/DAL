#!/usr/bin/env python3
"""Benchmark script comparing original and enhanced DAL pipelines."""

import time
from typing import Dict, Any, List, Tuple
import json
from pathlib import Path

from dal import run_pipeline, run_enhanced_pipeline


def time_function(func, *args, **kwargs) -> Tuple[Any, float]:
    """Time a function call and return the result and duration."""
    start_time = time.time()
    result = func(*args, **kwargs)
    duration = time.time() - start_time
    return result, duration


def run_benchmark(prompt: str, n_segments: int = 3) -> Dict[str, Any]:
    """Run both pipelines and collect metrics."""
    print(f"\n{'='*80}\nRunning benchmark with prompt:\n{prompt}\n{'='*80}")
    
    # Run original pipeline
    print("\nRunning original pipeline...")
    original_result, original_time = time_function(
        run_pipeline, prompt, n_segments=n_segments, expand=True
    )
    
    # Run enhanced pipeline
    print("\nRunning enhanced pipeline...")
    enhanced_result, enhanced_time = time_function(
        run_enhanced_pipeline, prompt, n_segments=n_segments, expand=True
    )
    
    # Calculate metrics
    metrics = {
        "prompt": prompt,
        "n_segments": n_segments,
        "original": {
            "time_seconds": original_time,
            "answer_length": len(original_result.get("answer", "")),
            "num_segments": len(original_result.get("segments", [])),
            "used_expansion": "expanded_segments" in original_result,
        },
        "enhanced": {
            "time_seconds": enhanced_time,
            "answer_length": len(enhanced_result.get("answer", "")),
            "num_segments": len(enhanced_result.get("segments", [])),
            "used_expansion": "expanded_segments" in enhanced_result,
        },
        "improvement": {
            "time_ratio": original_time / enhanced_time if enhanced_time > 0 else float('inf'),
            "length_ratio": enhanced_result.get("answer", "") and \
                           len(enhanced_result["answer"]) / max(1, len(original_result.get("answer", "a")))
        }
    }
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS:")
    print(f"Original time: {original_time:.2f}s")
    print(f"Enhanced time: {enhanced_time:.2f}s")
    print(f"Speedup: {metrics['improvement']['time_ratio']:.2f}x")
    print(f"Answer length (original/enhanced): {metrics['original']['answer_length']}/{metrics['enhanced']['answer_length']}")
    print("="*80 + "\n")
    
    return metrics


def main():
    """Run benchmarks with different prompts."""
    test_prompts = [
        "Explain the key principles of quantum computing and how it differs from classical computing",
        "Describe the causes and effects of climate change with examples",
        "Compare and contrast different machine learning algorithms",
    ]
    
    all_metrics = []
    
    for prompt in test_prompts:
        metrics = run_benchmark(prompt, n_segments=3)
        all_metrics.append(metrics)
    
    # Save results
    output_file = Path("benchmark_results_enhanced.json")
    with open(output_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nBenchmark results saved to {output_file}")


if __name__ == "__main__":
    main()

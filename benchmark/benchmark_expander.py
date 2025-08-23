#!/usr/bin/env python3
"""
Benchmark script for comparing ExpanderV2 performance before and after changes.

This script runs the DAL pipeline with both the old and new ExpanderV2 implementations
and compares their performance and output quality.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Add parent directory to path to import dal module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('benchmark_expander.log')
    ]
)
logger = logging.getLogger(__name__)

# Test cases
TEST_CASES = [
    {
        'name': 'short_question',
        'prompt': 'What is quantum mechanics?',
        'should_expand': False  # Should be skipped by the expander
    },
    {
        'name': 'medium_paragraph',
        'prompt': (
            'Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. '
            'It is the foundation of all quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum information science.'
        ),
        'should_expand': True
    },
    {
        'name': 'long_document',
        'prompt': (
            'Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. '
            'It is the foundation of all quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum information science. '
            'Classical physics, the description of physics that existed before theories of relativity and quantum mechanics, describes many aspects of nature at an ordinary (macroscopic) scale, '
            'while quantum mechanics explains the aspects of nature at small (atomic and subatomic) scales, for which classical mechanics is insufficient. '
            'Quantum mechanics differs from classical physics in that energy, momentum, angular momentum, and other quantities of a bound system are restricted to discrete values (quantization); '
            'objects have characteristics of both particles and waves (wave-particle duality); and there are limits to how accurately the value of a physical quantity can be predicted prior to its measurement, given a complete set of initial conditions (the uncertainty principle).'
        ),
        'should_expand': True
    }
]


def run_benchmark(test_cases: List[Dict], use_fixed: bool = True) -> Dict:
    """Run the benchmark with the specified expander version."""
    # Import the appropriate version of the expander
    if use_fixed:
        logger.info("Using FIXED ExpanderV2 implementation")
        from dal.expander_v2_fixed import ExpanderV2
    else:
        logger.info("Using ORIGINAL ExpanderV2 implementation")
        from dal.expander_v2 import ExpanderV2
    
    # Import the pipeline
    from dal.pipeline_v2 import run_enhanced_pipeline
    
    results = {}
    
    for test_case in test_cases:
        name = test_case['name']
        logger.info(f"\n{'='*80}")
        logger.info(f"Running test case: {name}")
        logger.info(f"Input length: {len(test_case['prompt'])} chars")
        
        try:
            # Run the pipeline
            start_time = time.time()
            result, artifact = run_enhanced_pipeline(
                prompt=test_case['prompt'],
                expand=True,
                return_artifact=True
            )
            duration = time.time() - start_time
            
            # Extract metrics
            metrics = {
                'duration_seconds': duration,
                'input_length': len(test_case['prompt']),
                'output_length': len(artifact.answer),
                'metrics': artifact.metrics
            }
            
            logger.info(f"Completed in {duration:.2f}s")
            logger.info(f"Input:  {test_case['prompt'][:100]}...")
            logger.info(f"Output: {artifact.answer[:200]}...")
            
            results[name] = {
                'success': True,
                'metrics': metrics,
                'input': test_case['prompt'],
                'output': artifact.answer
            }
            
        except Exception as e:
            logger.error(f"Error running test case {name}: {str(e)}", exc_info=True)
            results[name] = {
                'success': False,
                'error': str(e)
            }
    
    return results


def compare_results(original: Dict, fixed: Dict) -> Dict:
    """Compare results between original and fixed implementations."""
    comparison = {}
    
    for name in original.keys():
        if name not in fixed:
            continue
            
        orig_data = original[name]
        fixed_data = fixed[name]
        
        # Skip if either run failed
        if not orig_data['success'] or not fixed_data['success']:
            comparison[name] = {
                'comparable': False,
                'original_success': orig_data['success'],
                'fixed_success': fixed_data['success']
            }
            continue
        
        # Calculate comparison metrics
        orig_metrics = orig_data['metrics']
        fixed_metrics = fixed_data['metrics']
        
        comparison[name] = {
            'comparable': True,
            'duration_ratio': fixed_metrics['duration_seconds'] / orig_metrics['duration_seconds'],
            'output_length_ratio': fixed_metrics['output_length'] / orig_metrics['output_length'],
            'original_metrics': orig_metrics,
            'fixed_metrics': fixed_metrics
        }
    
    return comparison


def save_results(results: Dict, filename: str):
    """Save benchmark results to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {filename}")


def print_summary(comparison: Dict):
    """Print a summary of the benchmark comparison."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    for name, data in comparison.items():
        print(f"\nTest Case: {name}")
        print("-" * 60)
        
        if not data['comparable']:
            print(f"  Not comparable - Original: {data['original_success']}, Fixed: {data['fixed_success']}")
            continue
            
        print(f"  Duration Ratio (fixed/original): {data['duration_ratio']:.2f}x")
        print(f"  Output Length Ratio (fixed/original): {data['output_length_ratio']:.2f}x")
        
        # Print detailed metrics
        print("\n  Original Metrics:")
        print(f"    Duration: {data['original_metrics']['duration_seconds']:.2f}s")
        print(f"    Output Length: {data['original_metrics']['output_length']} chars")
        
        print("\n  Fixed Metrics:")
        print(f"    Duration: {data['fixed_metrics']['duration_seconds']:.2f}s")
        print(f"    Output Length: {data['fixed_metrics']['output_length']} chars")
        
        # Print pipeline metrics if available
        if 'metrics' in data['original_metrics']:
            print("\n  Original Pipeline Metrics:")
            print(f"    Expansion Time: {data['original_metrics']['metrics'].get('timing', {}).get('expansion_ms', 'N/A'):.1f}ms")
            print(f"    Expansion Ratio: {data['original_metrics']['metrics'].get('tokens', {}).get('expansion_ratio', 'N/A'):.2f}x")
            
            print("\n  Fixed Pipeline Metrics:")
            print(f"    Expansion Time: {data['fixed_metrics']['metrics'].get('timing', {}).get('expansion_ms', 'N/A'):.1f}ms")
            print(f"    Expansion Ratio: {data['fixed_metrics']['metrics'].get('tokens', {}).get('expansion_ratio', 'N/A'):.2f}x")


def main():
    """Run the benchmark and compare results."""
    parser = argparse.ArgumentParser(description='Benchmark ExpanderV2 implementations')
    parser.add_argument('--skip-original', action='store_true', help='Skip the original implementation benchmark')
    parser.add_argument('--skip-fixed', action='store_true', help='Skip the fixed implementation benchmark')
    args = parser.parse_args()
    
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    
    original_results = {}
    fixed_results = {}
    
    # Run benchmarks
    if not args.skip_original:
        logger.info("\n" + "="*80)
        logger.info("RUNNING ORIGINAL IMPLEMENTATION")
        logger.info("="*80)
        original_results = run_benchmark(TEST_CASES, use_fixed=False)
        save_results(original_results, results_dir / 'original_results.json')
    
    if not args.skip_fixed:
        logger.info("\n" + "="*80)
        logger.info("RUNNING FIXED IMPLEMENTATION")
        logger.info("="*80)
        fixed_results = run_benchmark(TEST_CASES, use_fixed=True)
        save_results(fixed_results, results_dir / 'fixed_results.json')
    
    # Compare results if both were run
    if original_results and fixed_results:
        comparison = compare_results(original_results, fixed_results)
        save_results(comparison, results_dir / 'comparison.json')
        print_summary(comparison)
    
    logger.info("\nBenchmark completed!")


if __name__ == "__main__":
    main()

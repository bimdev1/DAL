#!/usr/bin/env python3
"""Benchmark DAL against GPT-2 on factual synthesis tasks."""

import argparse
import datetime
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Local imports
import os
import sys
from pathlib import Path

# Add the benchmark directory to the path
benchmark_dir = Path(__file__).parent
if str(benchmark_dir) not in sys.path:
    sys.path.insert(0, str(benchmark_dir))

# Import local modules using absolute imports
from benchmark.dal_runner import generate_dal_response
from benchmark.gpt2_runner import generate_gpt2_response
from benchmark import evaluate_factual_synthesis

# Define utility functions locally to avoid import issues
def calculate_keyword_matches(text: str, keywords: List[str]) -> Tuple[List[str], float]:
    """Calculate which keywords are present in the text."""
    if not text or not keywords:
        return [], 0.0
        
    text_lower = text.lower()
    matched = [kw for kw in keywords if kw.lower() in text_lower]
    ratio = len(matched) / len(keywords) if keywords else 0.0
    return matched, ratio

def format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"

def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the benchmark."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('benchmark.log')
        ]
    )
    
    def format_duration(seconds: float) -> str:
        """Format duration in seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes)}m {seconds:.1f}s"

@dataclass
class BenchmarkResult:
    """Container for benchmark results with metrics."""
    prompt: str
    domain: str
    model: str
    response: str
    token_count: int
    matched_keywords: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    keyword_match_ratio: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

# Sample benchmark prompts for factual synthesis
BENCHMARK_PROMPTS = [
    {
        "id": "science_1",
        "prompt": "Summarize the key principles of quantum mechanics in simple terms.",
        "domain": "science",
        "expected_keywords": ["wave-particle", "uncertainty", "superposition", "quantization"]
    },
    {
        "id": "history_1",
        "prompt": "Explain the main causes of World War I.",
        "domain": "history",
        "expected_keywords": ["militarism", "alliances", "imperialism", "nationalism"]
    },
    {
        "id": "tech_1",
        "prompt": "Describe how blockchain technology works.",
        "domain": "technology",
        "expected_keywords": ["blocks", "decentralized", "hash", "transactions"]
    },
    {
        "id": "science_2",
        "prompt": "What are the main differences between DNA and RNA?",
        "domain": "science",
        "expected_keywords": ["deoxyribose", "ribose", "thymine", "uracil", "double-stranded", "single-stranded"]
    },
    {
        "id": "history_2",
        "prompt": "What were the key events of the American Civil Rights Movement?",
        "domain": "history",
        "expected_keywords": ["segregation", "Martin Luther King", "Civil Rights Act", "voting rights"]
    }
]

def run_benchmark(
    prompts: List[Dict[str, Any]],
    max_tokens: int = 150,
    skip_gpt2: bool = False,
    verbose: bool = False
) -> List[BenchmarkResult]:
    """Run benchmark on a list of prompts.
    
    Args:
        prompts: List of prompt dictionaries with 'prompt', 'domain', and 'expected_keywords'
        max_tokens: Maximum number of tokens to generate per response
        skip_gpt2: If True, only run DAL benchmark
        verbose: Enable verbose output
        
    Returns:
        List of BenchmarkResult objects with detailed metrics
    """
    results = []
    
    for i, prompt_data in enumerate(prompts, 1):
        prompt = prompt_data["prompt"]
        domain = prompt_data.get("domain", "unknown")
        expected_keywords = prompt_data.get("expected_keywords", [])
        
        logging.info(f"\nProcessing prompt {i}/{len(prompts)}")
        logging.info(f"Domain: {domain}")
        if expected_keywords:
            logging.info(f"Expected keywords: {', '.join(expected_keywords)}")
        
        if verbose:
            print("\n" + "="*80)
            print(f"PROMPT: {prompt}")
            print(f"DOMAIN: {domain}")
            if expected_keywords:
                print(f"EXPECTED KEYWORDS: {', '.join(expected_keywords)}")
            print("="*80 + "\n")
        
        # Test DAL
        logging.info("Testing DAL...")
        try:
            dal_result = evaluate_factual_synthesis(
                prompt=prompt,
                model="dal",
                max_tokens=max_tokens,
                domain=domain,
                expected_keywords=expected_keywords
            )
            
            dal_benchmark = BenchmarkResult(
                prompt=prompt,
                domain=domain,
                model="dal_v3",
                response=dal_result.get("response", ""),
                token_count=dal_result.get("token_count", 0),
                matched_keywords=dal_result.get("matched_keywords", []),
                expected_keywords=expected_keywords,
                keyword_match_ratio=dal_result.get("keyword_match_ratio", 0.0),
                metrics=dal_result.get("metrics", {})
            )
            results.append(dal_benchmark)
            
            if verbose:
                print(f"\nDAL Response ({dal_benchmark.token_count} tokens, {len(dal_benchmark.matched_keywords)}/{len(expected_keywords)} keywords):")
                print("-" * 40)
                print(dal_benchmark.response)
                if dal_benchmark.matched_keywords:
                    print(f"\n✓ Matched keywords: {', '.join(dal_benchmark.matched_keywords)}")
            
            logging.info(f"DAL completed: {dal_benchmark.token_count} tokens, "
                        f"{len(dal_benchmark.matched_keywords)}/{len(expected_keywords)} keywords matched")
            
        except Exception as e:
            logging.error(f"Error running DAL benchmark: {str(e)}", exc_info=True)
            if verbose:
                print(f"\n❌ Error running DAL benchmark: {str(e)}")
        
        # Test GPT-2 if not skipped
        if not skip_gpt2:
            logging.info("Testing GPT-2...")
            try:
                gpt2_result = evaluate_factual_synthesis(
                    prompt=prompt,
                    model="gpt2",
                    max_tokens=max_tokens,
                    domain=domain,
                    expected_keywords=expected_keywords
                )
                
                gpt2_benchmark = BenchmarkResult(
                    prompt=prompt,
                    domain=domain,
                    model="gpt2",
                    response=gpt2_result.get("response", ""),
                    token_count=gpt2_result.get("token_count", 0),
                    matched_keywords=gpt2_result.get("matched_keywords", []),
                    expected_keywords=expected_keywords,
                    keyword_match_ratio=gpt2_result.get("keyword_match_ratio", 0.0),
                    metrics=gpt2_result.get("metrics", {})
                )
                results.append(gpt2_benchmark)
                
                if verbose:
                    print(f"\nGPT-2 Response ({gpt2_benchmark.token_count} tokens, {len(gpt2_benchmark.matched_keywords)}/{len(expected_keywords)} keywords):")
                    print("-" * 40)
                    print(gpt2_benchmark.response)
                    if gpt2_benchmark.matched_keywords:
                        print(f"\n✓ Matched keywords: {', '.join(gpt2_benchmark.matched_keywords)}")
                
                logging.info(f"GPT-2 completed: {gpt2_benchmark.token_count} tokens, "
                            f"{len(gpt2_benchmark.matched_keywords)}/{len(expected_keywords)} keywords matched")
                
            except Exception as e:
                logging.error(f"Error running GPT-2 benchmark: {str(e)}", exc_info=True)
                if verbose:
                    print(f"\n❌ Error running GPT-2 benchmark: {str(e)}")
        
        if verbose:
            print("\n" + "="*80)
    
    return results

def analyze_results(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Analyze benchmark results with detailed metrics.
    
    Args:
        results: List of BenchmarkResult objects
        
    Returns:
        Dictionary containing aggregated metrics and analysis
    """
    metrics = {
        "total_questions": len([r for r in results if r.model == "dal_v3"]),
        "models": {
            "dal_v3": {
                "total_responses": 0,
                "total_tokens": 0,
                "total_keywords": 0,
                "domains": {}
            },
            "gpt2": {
                "total_responses": 0,
                "total_tokens": 0,
                "total_keywords": 0,
                "domains": {}
            }
        },
        "domains": {}
    }
    
    # Initialize domain tracking
    all_domains = set(r.domain.lower() for r in results)
    for domain in all_domains:
        metrics["domains"][domain] = {
            "count": 0,
            "dal_v3": {"total_responses": 0, "total_tokens": 0, "total_keywords": 0},
            "gpt2": {"total_responses": 0, "total_tokens": 0, "total_keywords": 0}
        }
    
    # Process each result
    for result in results:
        model_key = result.model.lower()
        domain = result.domain.lower()
        
        # Initialize domain model if not exists
        if domain not in metrics["models"][model_key]["domains"]:
            metrics["models"][model_key]["domains"][domain] = {
                "total_responses": 0,
                "total_tokens": 0,
                "total_keywords": 0
            }
        
        # Update model metrics
        metrics["models"][model_key]["total_responses"] += 1
        metrics["models"][model_key]["total_tokens"] += result.token_count
        metrics["models"][model_key]["total_keywords"] += len(result.matched_keywords)
        
        # Update domain-specific metrics
        metrics["models"][model_key]["domains"][domain]["total_responses"] += 1
        metrics["models"][model_key]["domains"][domain]["total_tokens"] += result.token_count
        metrics["models"][model_key]["domains"][domain]["total_keywords"] += len(result.matched_keywords)
        
        # Update domain metrics
        metrics["domains"][domain][model_key]["total_responses"] += 1
        metrics["domains"][domain][model_key]["total_tokens"] += result.token_count
        metrics["domains"][domain][model_key]["total_keywords"] += len(result.matched_keywords)
        metrics["domains"][domain]["count"] = metrics["domains"][domain]["dal_v3"]["total_responses"] + metrics["domains"][domain]["gpt2"]["total_responses"]
    
    # Calculate averages for models
    for model in metrics["models"]:
        m = metrics["models"][model]
        if m["total_responses"] > 0:
            m["avg_response_length"] = round(m["total_tokens"] / m["total_responses"], 2)
            m["avg_keyword_matches"] = round(m["total_keywords"] / m["total_responses"], 2)
            m["keywords_per_token"] = round(m["total_keywords"] / m["total_tokens"], 4) if m["total_tokens"] > 0 else 0.0
        
        # Calculate averages for each domain in this model
        for domain, domain_data in m["domains"].items():
            if domain_data["total_responses"] > 0:
                domain_data["avg_response_length"] = round(domain_data["total_tokens"] / domain_data["total_responses"], 2)
                domain_data["avg_keyword_matches"] = round(domain_data["total_keywords"] / domain_data["total_responses"], 2)
                domain_data["keywords_per_token"] = round(domain_data["total_keywords"] / domain_data["total_tokens"], 4) if domain_data["total_tokens"] > 0 else 0.0
    
    # Calculate domain-level averages
    for domain, domain_data in metrics["domains"].items():
        for model in ["dal_v3", "gpt2"]:
            m = domain_data[model]
            if m["total_responses"] > 0:
                m["avg_response_length"] = round(m["total_tokens"] / m["total_responses"], 2)
                m["avg_keyword_matches"] = round(m["total_keywords"] / m["total_responses"], 2)
                m["keywords_per_token"] = round(m["total_keywords"] / m["total_tokens"], 4) if m["total_tokens"] > 0 else 0.0
    
    return metrics

def save_results(results: Any, output_file: str) -> None:
    """Save benchmark results to a JSON file.
    
    Args:
        results: Results to save (list of BenchmarkResult or dict)
        output_file: Path to output file
    """
    # Convert dataclasses to dictionaries if needed
    if isinstance(results, list) and results and hasattr(results[0], '__dataclass_fields__'):
        results = [asdict(r) for r in results]
    elif hasattr(results, '__dataclass_fields__'):
        results = asdict(results)
    
    # Ensure the directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save with indentation for readability
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {os.path.abspath(output_file)}")

def safe_get(dictionary, *keys, default=0):
    """Safely get a value from a nested dictionary with a default if any key is missing."""
    for key in keys:
        try:
            dictionary = dictionary[key]
        except (KeyError, TypeError):
            return default
    return dictionary

def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print metrics in a clear, organized format.
    
    Args:
        metrics: Dictionary containing benchmark metrics
    """
    if not metrics:
        print("No metrics data available to display.")
        return
        
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    # Overall summary
    print("\nOVERALL SUMMARY")
    print("-"*40)
    print(f"Total questions: {safe_get(metrics, 'total_questions', default='N/A')}")
    
    # Model comparison
    print("\nMODEL COMPARISON")
    print("-"*80)
    print(f"{'METRIC':<30} {'DAL v3':<20} {'GPT-2':<20}")
    print("-"*80)
    
    # Get model data with safe defaults
    dal_metrics = safe_get(metrics, 'models', 'dal_v3', default={})
    gpt2_metrics = safe_get(metrics, 'models', 'gpt2', default={})
    
    # Response length
    dal_len = safe_get(dal_metrics, 'avg_response_length')
    gpt2_len = safe_get(gpt2_metrics, 'avg_response_length')
    print(f"{'Avg. Response Length':<30} {dal_len:<20.1f} {gpt2_len:<20.1f}")
    
    # Keyword matches
    dal_kw = safe_get(dal_metrics, 'avg_keyword_matches')
    gpt2_kw = safe_get(gpt2_metrics, 'avg_keyword_matches')
    print(f"{'Avg. Keyword Matches':<30} {dal_kw:<20.1f} {gpt2_kw:<20.1f}")
    
    # Keywords per token (efficiency)
    dal_kpt = safe_get(dal_metrics, 'keywords_per_token', default=0) * 100
    gpt2_kpt = safe_get(gpt2_metrics, 'keywords_per_token', default=0) * 100
    print(f"{'Keywords per Token (x100)':<30} {dal_kpt:<20.2f} {gpt2_kpt:<20.2f}")
    
    # Total keywords matched
    dal_total = safe_get(dal_metrics, 'total_keywords')
    gpt2_total = safe_get(gpt2_metrics, 'total_keywords')
    print(f"{'Total Keywords Matched':<30} {dal_total:<20} {gpt2_total:<20}")
    
    # Domain-specific metrics
    print("\nDOMAIN-SPECIFIC METRICS")
    print("-"*80)
    
    for domain, data in metrics['domains'].items():
        if domain == 'count':
            continue
            
        print(f"\n{domain.upper()} (n={data['count']//2} questions per model)")
        print("-" * 60)
        print(f"{'METRIC':<25} {'DAL v3':<15} {'GPT-2':<15}")
        print("-"*60)
        
        # Response length
        dal_len = data.get('dal_v3', {}).get('avg_response_length', 0)
        gpt2_len = data.get('gpt2', {}).get('avg_response_length', 0)
        print(f"{'Avg. Response Length':<25} {dal_len:<15.1f} {gpt2_len:<15.1f}")
        
        # Keyword matches
        dal_kw = data.get('dal_v3', {}).get('avg_keyword_matches', 0)
        gpt2_kw = data.get('gpt2', {}).get('avg_keyword_matches', 0)
        print(f"{'Avg. Keyword Matches':<25} {dal_kw:<15.1f} {gpt2_kw:<15.1f}")
        
        # Keywords per token (efficiency)
        dal_kpt = data.get('dal_v3', {}).get('keywords_per_token', 0) * 100
        gpt2_kpt = data.get('gpt2', {}).get('keywords_per_token', 0) * 100
        print(f"{'Keywords per Token (x100)':<25} {dal_kpt:<15.2f} {gpt2_kpt:<15.2f}")
        
        # Total keywords
        dal_total = safe_get(data, 'dal_v3', 'total_keywords', default=0)
        gpt2_total = safe_get(data, 'gpt2', 'total_keywords', default=0)
        print(f"{'Total Keywords Matched':<25} {dal_total:<15} {gpt2_total:<15}")
    
    # Detailed model performance
    print("\n" + "="*80)
    print("DETAILED MODEL PERFORMANCE")
    print("-"*80)
    
    # DAL v3 performance
    print("\nDAL_V3 PERFORMANCE")
    print("-"*40)
    model_data = safe_get(metrics, 'models', 'dal_v3', default={})
    print(f"Total Responses: {safe_get(model_data, 'total_responses', default=0)}")
    print(f"Total Tokens: {safe_get(model_data, 'total_tokens', default=0)}")
    print(f"Total Keywords Matched: {safe_get(model_data, 'total_keywords', default=0)}")
    print(f"Avg. Response Length: {safe_get(model_data, 'avg_response_length', default=0.0):.1f} tokens")
    print(f"Avg. Keyword Matches: {safe_get(model_data, 'avg_keyword_matches', default=0.0):.2f} per response")
    
    # GPT-2 performance
    print("\nGPT-2 PERFORMANCE")
    print("-"*40)
    model_data = safe_get(metrics, 'models', 'gpt2', default={})
    print(f"Total Responses: {safe_get(model_data, 'total_responses', default=0)}")
    print(f"Total Tokens: {safe_get(model_data, 'total_tokens', default=0)}")
    print(f"Total Keywords Matched: {safe_get(model_data, 'total_keywords', default=0)}")
    print(f"Avg. Response Length: {safe_get(model_data, 'avg_response_length', default=0.0):.1f} tokens")
    print(f"Avg. Keyword Matches: {safe_get(model_data, 'avg_keyword_matches', default=0.0):.2f} per response")

def main():
    """Run the benchmark with command-line configuration.
    
    This function handles command-line arguments, runs the benchmark, and saves the results.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Benchmark DAL v3 vs. GPT-2 on factual synthesis tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add command-line arguments
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate per response"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=[],
        help="Filter prompts by these domains (default: all domains)"
    )
    parser.add_argument(
        "--skip-gpt2",
        action="store_true",
        help="Skip GPT-2 benchmarking (only run DAL v3)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output during benchmark execution"
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "benchmark.log"), mode='w')
        ]
    )
    
    # Filter prompts by domain if specified
    if args.domains:
        filtered_prompts = [
            p for p in BENCHMARK_PROMPTS 
            if p.get("domain", "").lower() in [d.lower() for d in args.domains]
        ]
        if not filtered_prompts:
            logging.error(f"No prompts found for domains: {', '.join(args.domains)}")
            return 1
    else:
        filtered_prompts = BENCHMARK_PROMPTS
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for output files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"benchmark_results_{timestamp}.json")
    
    # Print benchmark configuration
    print("\n" + "="*80)
    print("BENCHMARK CONFIGURATION")
    print("="*80)
    print(f"• Models: {'DAL v3' + (' and GPT-2' if not args.skip_gpt2 else '')}")
    print(f"• Domains: {', '.join(args.domains) if args.domains else 'All'}")
    print(f"• Max tokens per response: {args.max_tokens}")
    print(f"• Output directory: {os.path.abspath(args.output_dir)}")
    print("="*80 + "\n")
    
    try:
        # Run the benchmark
        logging.info(f"Starting benchmark with {len(filtered_prompts)} prompts...")
        start_time = time.time()
        
        results = run_benchmark(
            prompts=filtered_prompts,
            max_tokens=args.max_tokens,
            skip_gpt2=args.skip_gpt2
        )
        
        # Calculate benchmark duration
        duration = time.time() - start_time
        
        # Analyze results
        logging.info("Analyzing results...")
        metrics = analyze_results(results)
        
        # Print summary metrics
        print_metrics(metrics)
        
        # Save results
        logging.info(f"Saving results to {output_file}")
        save_results(metrics, output_file)
        
        # Save raw results for further analysis
        raw_output = os.path.join(args.output_dir, f"raw_results_{timestamp}.json")
        save_results(results, raw_output)
        
        # Print completion message
        print("\n" + "="*80)
        print(f"BENCHMARK COMPLETED IN {duration:.1f} SECONDS")
        print("="*80)
        print(f"Results saved to: {os.path.abspath(output_file)}")
        print(f"Raw data saved to: {os.path.abspath(raw_output)}")
        print("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        logging.warning("Benchmark interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Benchmark failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    main()

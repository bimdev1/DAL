#!/usr/bin/env python3
"""Run benchmark on philosophy prompts using DAL v3."""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('philosophy_benchmark.log')
    ]
)
logger = logging.getLogger(__name__)

# Import local modules
import sys
from pathlib import Path

# Add the benchmark directory to the path
benchmark_dir = Path(__file__).parent
if str(benchmark_dir) not in sys.path:
    sys.path.insert(0, str(benchmark_dir))

from benchmark.dal_runner import generate_dal_response
from benchmark.gpt2_runner import generate_gpt2_response

# Philosophy prompts for benchmarking
PHILOSOPHY_PROMPTS = [
    {
        "id": "phil_1",
        "prompt": """# Truth vs Stability in Society

Analyze what it means for a society to value truth over stability. Your response should:
1. Define what 'valuing truth' and 'stability' mean in a societal context
2. Provide historical examples of societies that prioritized truth over stability
3. Discuss the potential benefits and drawbacks of this prioritization
4. Consider how this choice affects different aspects of society (politics, education, media)
5. Conclude with whether you believe this prioritization is sustainable long-term""",
        "domain": "philosophy",
        "expected_keywords": ["truth", "stability", "society", "ethics", "governance"],
        "min_length": 300
    },
    {
        "id": "phil_2",
        "prompt": """# The Trolley Problem Revisited

Examine the ethical implications of the classic trolley problem from multiple philosophical perspectives. Your response should:
1. Explain the original trolley problem and its variations
2. Analyze the utilitarian perspective on this dilemma
3. Discuss deontological objections to the utilitarian approach
4. Explore how virtue ethics would approach this problem
5. Consider the implications for autonomous vehicle programming""",
        "domain": "philosophy",
        "expected_keywords": ["trolley problem", "ethics", "utilitarianism", "deontology", "virtue ethics"],
        "min_length": 350
    },
    {
        "id": "phil_3",
        "prompt": """# The Nature of Consciousness

Explore different philosophical theories of consciousness. Your response should:
1. Define consciousness from a philosophical perspective
2. Explain the mind-body problem and its significance
3. Compare and contrast dualism and physicalism
4. Discuss the hard problem of consciousness as proposed by David Chalmers
5. Evaluate the implications for artificial intelligence""",
        "domain": "philosophy",
        "expected_keywords": ["consciousness", "mind-body problem", "dualism", "physicalism", "qualia"],
        "min_length": 400
    },
    {
        "id": "phil_4",
        "prompt": """# Free Will vs Determinism

Analyze the philosophical debate between free will and determinism. Your response should:
1. Define free will and determinism
2. Explain the arguments for hard determinism
3. Discuss compatibilist perspectives
4. Explore libertarian free will
5. Consider the implications for moral responsibility""",
        "domain": "philosophy",
        "expected_keywords": ["free will", "determinism", "compatibilism", "moral responsibility", "libertarianism"],
        "min_length": 350
    },
    {
        "id": "phil_5",
        "prompt": """# The Ethics of Artificial Intelligence

Examine the ethical considerations in developing advanced AI systems. Your response should:
1. Define AI ethics and its importance
2. Discuss the alignment problem
3. Analyze the concept of AI rights
4. Explore potential existential risks
5. Propose ethical guidelines for AI development""",
        "domain": "philosophy",
        "expected_keywords": ["AI ethics", "alignment problem", "machine rights", "existential risk", "superintelligence"],
        "min_length": 300
    }
]

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    prompt_id: str
    prompt: str
    domain: str
    model: str
    response: str
    token_count: int
    matched_keywords: List[str]
    expected_keywords: List[str]
    keyword_match_ratio: float
    metrics: Dict[str, Any]
    generation_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'prompt_id': self.prompt_id,
            'prompt': self.prompt,
            'domain': self.domain,
            'model': self.model,
            'response': self.response,
            'token_count': self.token_count,
            'matched_keywords': self.matched_keywords,
            'expected_keywords': self.expected_keywords,
            'keyword_match_ratio': self.keyword_match_ratio,
            'generation_time': self.generation_time,
            'metrics': self.metrics
        }

def calculate_keyword_matches(text: str, keywords: List[str]) -> Tuple[List[str], float]:
    """Calculate which keywords are present in the text."""
    if not text or not keywords:
        return [], 0.0
        
    text_lower = text.lower()
    matched = [kw for kw in keywords if kw.lower() in text_lower]
    ratio = len(matched) / len(keywords) if keywords else 0.0
    return matched, ratio

def run_benchmark(
    prompts: List[Dict[str, Any]],
    max_tokens: int = 400,
    skip_gpt2: bool = False,
    verbose: bool = False
) -> List[BenchmarkResult]:
    """Run benchmark on philosophy prompts.
    
    Args:
        prompts: List of prompt dictionaries
        max_tokens: Maximum tokens to generate
        skip_gpt2: If True, skip GPT-2 baseline
        verbose: Enable verbose output
        
    Returns:
        List of BenchmarkResult objects
    """
    results = []
    
    for prompt in prompts:
        logger.info(f"\n{'='*80}")
        logger.info(f"PROMPT: {prompt['id']} - {prompt['prompt'][:100]}...")
        
        # Run DAL v3
        try:
            logger.info("\n[RUNNING DAL v3]")
            start_time = time.time()
            
            dal_result = generate_dal_response(
                prompt=prompt['prompt'],
                max_tokens=max_tokens
            )
            
            gen_time = time.time() - start_time
            dal_response = dal_result.get('response', '')
            tokens = dal_result.get('token_count', len(dal_response.split()))
            
            matched, ratio = calculate_keyword_matches(
                dal_response, prompt['expected_keywords']
            )
            
            result = BenchmarkResult(
                prompt_id=prompt['id'],
                prompt=prompt['prompt'],
                domain=prompt['domain'],
                model='DAL_v3',
                response=dal_response,
                token_count=tokens,
                matched_keywords=matched,
                expected_keywords=prompt['expected_keywords'],
                keyword_match_ratio=ratio,
                metrics={},
                generation_time=gen_time
            )
            
            results.append(result)
            logger.info(f"DAL completed: {tokens} tokens, {len(matched)}/{len(prompt['expected_keywords'])} keywords matched")
            
            # Run GPT-2 baseline if not skipped
            if not skip_gpt2:
                logger.info("\n[RUNNING GPT-2]")
                start_time = time.time()
                
                gpt2_response = generate_gpt2_response(
                    prompt=prompt['prompt'],
                    max_tokens=max_tokens
                )
                
                gen_time = time.time() - start_time
                tokens = len(gpt2_response.split())  # Simple token estimation
                
                matched, ratio = calculate_keyword_matches(
                    gpt2_response, prompt['expected_keywords']
                )
                
                gpt2_result = BenchmarkResult(
                    prompt_id=prompt['id'],
                    prompt=prompt['prompt'],
                    domain=prompt['domain'],
                    model='GPT-2',
                    response=gpt2_response,
                    token_count=tokens,
                    matched_keywords=matched,
                    expected_keywords=prompt['expected_keywords'],
                    keyword_match_ratio=ratio,
                    metrics={},
                    generation_time=gen_time
                )
                
                results.append(gpt2_result)
                logger.info(f"GPT-2 completed: {tokens} tokens, {len(matched)}/{len(prompt['expected_keywords'])} keywords matched")
                
        except Exception as e:
            logger.error(f"Error processing prompt {prompt['id']}: {str(e)}", exc_info=True)
            continue
    
    return results

def analyze_results(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Analyze and summarize benchmark results."""
    if not results:
        return {}
    
    # Group by model
    by_model = {}
    for result in results:
        if result.model not in by_model:
            by_model[result.model] = []
        by_model[result.model].append(result)
    
    # Calculate metrics
    metrics = {}
    for model, model_results in by_model.items():
        total_tokens = sum(r.token_count for r in model_results)
        total_keywords = sum(len(r.matched_keywords) for r in model_results)
        total_expected = sum(len(r.expected_keywords) for r in model_results)
        avg_time = sum(r.generation_time for r in model_results) / len(model_results)
        
        metrics[model] = {
            'total_responses': len(model_results),
            'total_tokens': total_tokens,
            'avg_tokens': total_tokens / len(model_results) if model_results else 0,
            'total_keywords_matched': total_keywords,
            'avg_keywords_per_response': total_keywords / len(model_results) if model_results else 0,
            'keyword_match_rate': total_keywords / total_expected if total_expected else 0,
            'avg_generation_time': avg_time,
            'tokens_per_second': total_tokens / sum(r.generation_time for r in model_results) if model_results else 0
        }
    
    return metrics

def save_results(results: List[BenchmarkResult], output_dir: str = 'benchmark_results') -> None:
    """Save benchmark results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results
    results_file = os.path.join(output_dir, f'philosophy_results_{timestamp}.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
    
    # Generate and save summary
    summary = {
        'timestamp': timestamp,
        'total_prompts': len(set(r.prompt_id for r in results)),
        'metrics': analyze_results(results),
        'results_file': results_file
    }
    
    summary_file = os.path.join(output_dir, f'philosophy_summary_{timestamp}.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to:")
    logger.info(f"- Detailed results: {results_file}")
    logger.info(f"- Summary: {summary_file}")

def main():
    """Run the benchmark with command-line arguments."""
    parser = argparse.ArgumentParser(description='Run philosophy benchmark with DAL v3')
    parser.add_argument('--max-tokens', type=int, default=400, help='Maximum tokens to generate per response')
    parser.add_argument('--skip-gpt2', action='store_true', help='Skip GPT-2 baseline')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--output-dir', default='benchmark_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    logger.info("Starting philosophy benchmark...")
    logger.info(f"Using max_tokens={args.max_tokens}, skip_gpt2={args.skip_gpt2}")
    
    # Run benchmark
    start_time = time.time()
    results = run_benchmark(
        PHILOSOPHY_PROMPTS,
        max_tokens=args.max_tokens,
        skip_gpt2=args.skip_gpt2,
        verbose=args.verbose
    )
    
    # Save results
    save_results(results, args.output_dir)
    
    # Print summary
    metrics = analyze_results(results)
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*80)
    
    for model, stats in metrics.items():
        logger.info(f"\n{model}:")
        logger.info(f"  Responses: {stats['total_responses']}")
        logger.info(f"  Avg. tokens/response: {stats['avg_tokens']:.1f}")
        logger.info(f"  Total keywords matched: {stats['total_keywords_matched']}")
        logger.info(f"  Keyword match rate: {stats['keyword_match_rate']*100:.1f}%")
        logger.info(f"  Avg. generation time: {stats['avg_generation_time']:.2f}s")
        logger.info(f"  Tokens/second: {stats['tokens_per_second']:.1f}")
    
    logger.info("\n" + "="*80)
    logger.info(f"Benchmark completed in {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    main()

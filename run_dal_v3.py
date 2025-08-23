#!/usr/bin/env python3
"""
DAL v3 Runner with Local SDT-Aware Generation

This script demonstrates how to use the enhanced DAL pipeline with local model generation
that respects Segment Descriptor Tags (SDTs) for fine-grained control over text generation.
"""

import argparse
import json
import logging
import time
from typing import Dict, Any, Optional

from dal import (
    run_enhanced_pipeline_v3,
    PipelineConfig,
    LocalGenerator,
    get_version,
    log_versions
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run DAL v3 with local generation')
    
    # Input/output
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input text or @filename to read from (default: stdin)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file (default: stdout)'
    )
    
    # Pipeline configuration
    parser.add_argument(
        '--segments', '-n',
        type=int,
        default=3,
        help='Target number of segments (default: 3)'
    )
    parser.add_argument(
        '--show-vectors',
        action='store_true',
        help='Include vector representations in output'
    )
    
    # Local generation settings
    local_group = parser.add_argument_group('Local Generation')
    local_group.add_argument(
        '--use-local',
        action='store_true',
        help='Enable local model generation'
    )
    local_group.add_argument(
        '--model',
        type=str,
        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        help='Model name or path (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)'
    )
    local_group.add_argument(
        '--backend',
        type=str,
        choices=['huggingface', 'gguf'],
        default='huggingface',
        help='Backend to use for local generation (default: huggingface)'
    )
    local_group.add_argument(
        '--max-tokens',
        type=int,
        default=128,
        help='Maximum number of tokens to generate per segment (default: 128)'
    )
    local_group.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (0.0-1.0, default: 0.7)'
    )
    local_group.add_argument(
        '--timeout',
        type=float,
        default=8.0,
        help='Timeout in seconds for generation (default: 8.0)'
    )
    local_group.add_argument(
        '--sdt-control',
        action='store_true',
        help='Respect SDT tags for generation control'
    )
    local_group.add_argument(
        '--force-regen',
        action='store_true',
        help='Force regeneration of all segments (overrides SDT control)'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Increase verbosity (can be used multiple times)'
    )
    
    return parser.parse_args()

def get_input_text(args: argparse.Namespace) -> str:
    """Get input text from file or command line."""
    if args.input and args.input.startswith('@'):
        try:
            with open(args.input[1:], 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            raise
    return args.input or input("Enter your prompt: ")

def write_output(output: str, args: argparse.Namespace) -> None:
    """Write output to file or console."""
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            logger.info(f"Output written to {args.output}")
        except Exception as e:
            logger.error(f"Error writing output file: {e}")
            raise
    else:
        print("\n" + "=" * 80)
        print(output)
        print("=" * 80 + "\n")

def setup_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level."""
    log_level = logging.WARNING
    if verbosity == 1:
        log_level = logging.INFO
    elif verbosity >= 2:
        log_level = logging.DEBUG
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def format_metrics(metrics: Dict[str, Any]) -> str:
    """Format metrics for display."""
    lines = ["\n=== Generation Metrics ==="]
    
    # Timing information
    timing = metrics.get('timing', {})
    if timing:
        lines.append("\nTiming (seconds):")
        for key, value in timing.items():
            lines.append(f"  {key.replace('_', ' ').title()}: {value:.4f}")
    
    # Token counts
    tokens = metrics.get('tokens', {})
    if tokens:
        lines.append("\nTokens:")
        for key, value in tokens.items():
            lines.append(f"  {key.replace('_', ' ').title()}: {value:,}")
    
    # Generation stats
    gen_metrics = metrics.get('generation', {})
    if gen_metrics:
        lines.append("\nGeneration:")
        lines.append(f"  Segments Processed: {gen_metrics.get('segments_processed', 0)}")
        lines.append(f"  Segments Regenerated: {gen_metrics.get('segments_regenerated', 0)}")
        lines.append(f"  Total Tokens In: {gen_metrics.get('total_tokens_in', 0):,}")
        lines.append(f"  Total Tokens Out: {gen_metrics.get('total_tokens_out', 0):,}")
        lines.append(f"  Total Time: {gen_metrics.get('total_time', 0):.2f}s")
        
        # Per-segment details
        if gen_metrics.get('per_segment'):
            lines.append("\n  Per-segment details:")
            for seg in gen_metrics['per_segment']:
                if seg.get('regenerated'):
                    lines.append(
                        f"  - Segment {seg['index'] + 1}: "
                        f"{seg['tokens_in']} → {seg['tokens_out']} tokens "
                        f"({seg['time_taken']:.2f}s, {seg['backend']} - {seg['model']})"
                    )
    
    return "\n".join(lines)

def main() -> None:
    """Main function to run the DAL v3 pipeline."""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    # Log version information
    logger.info("DAL v3 Runner - %s", get_version())
    log_versions()
    
    try:
        # Get input text
        input_text = get_input_text(args)
        if not input_text:
            raise ValueError("No input text provided")
            
        # Prepare model configuration
        model_spec = {
            'backend': args.backend,
            'model_name_or_path': args.model,
            'device_map': 'auto',
            'load_in_4bit': True,
        }
        
        # Configure pipeline
        config = PipelineConfig(
            n_segments=args.segments,
            show_vectors=args.show_vectors,
            use_local_generation=args.use_local,
            sdt_control=args.sdt_control,
            force_regen=args.force_regen,
            local_model_spec=model_spec,
            max_new_tokens=args.max_tokens,
            local_generation_timeout=args.timeout
        )
        
        # Run the pipeline
        start_time = time.time()
        logger.info("Running pipeline...")
        
        result, artifact = run_enhanced_pipeline_v3(
            input_text,
            config=config,
            return_artifact=True
        )
        
        # Prepare output
        output = {
            'input': input_text,
            'output': result['text'],
            'segments': result['segments'],
            'sdts': result['sdts'],
            'metrics': result['metrics']
        }
        
        # Add vectors if requested
        if args.show_vectors and 'vectors' in result:
            output['vectors'] = result['vectors']
        
        # Convert to JSON for output
        output_json = json.dumps(output, indent=2, ensure_ascii=False)
        
        # Write output
        write_output(output_json, args)
        
        # Print metrics summary
        if args.verbose >= 1:
            print(format_metrics(result['metrics']))
        
        logger.info("Pipeline completed in %.2f seconds", time.time() - start_time)
        
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=args.verbose >= 1)
        raise

if __name__ == "__main__":
    main()

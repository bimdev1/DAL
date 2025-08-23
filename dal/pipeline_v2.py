"""Enhanced DAL pipeline with dynamic processing.

This version includes improvements from DAL Patch 2, such as dynamic
segmentation and adaptive generation parameters. It's designed to work
with ExpanderV2 for better handling of various input lengths and more
consistent output quality. It also integrates the DAL schema for rich
metadata and export capabilities.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Union, Tuple

from .segmenter import segment_prompt
from .tagging import assign_tags
from .stitcher import stitch_segments
from .vectorizer import simple_vectorise
from .dal_types import DALTagBlock, DALVector, DALBlock, DALRunArtifact
import logging

logger = logging.getLogger(__name__)


def _extract_primary_intent(prompt: str) -> str:
    """Extract primary intent from prompt (simplified for now)."""
    # TODO: Replace with more sophisticated intent extraction
    return prompt[:100] + ('...' if len(prompt) > 100 else '')


def _create_sdt_from_tags(tags: List[str]) -> DALTagBlock:
    """Create a Synthetic DAL Tag (SDT) block from a list of tags."""
    # Map common tags to SDT fields
    sdt = DALTagBlock()
    
    # Simple mapping - can be expanded based on your tag structure
    for tag in tags:
        if ':' in tag:
            key, value = tag.split(':', 1)
            if hasattr(sdt, key):
                try:
                    setattr(sdt, key, type(getattr(sdt, key))(value))
                except (ValueError, TypeError):
                    pass  # Keep default if type conversion fails
    
    return sdt


def run_enhanced_pipeline(
    prompt: str,
    n_segments: int = 3,
    *,
    expand: bool = False,
    show_vectors: bool = False,
    expander_model: Optional[str] = None,
    return_artifact: bool = False,
) -> Union[Dict[str, object], Tuple[Dict[str, object], DALRunArtifact]]:
    """
    Run the enhanced DAL pipeline with optional expansion.
    
    - Adaptive generation parameters
    - Better error handling and fallbacks
    - Support for the improved ExpanderV2
    - Version-aware execution with compatibility checks

    Args:
        prompt: The natural language prompt to process.
        n_segments: The maximum number of segments to use (actual number may be
                   lower based on input length).
        expand: If True, use ExpanderV2 to enhance each segment.
        show_vectors: If True, include vector representations in the output.
        expander_model: Optional model name for the expander.
        return_artifact: If True, return a DALRunArtifact with rich metadata.
        
    Returns:
        Dictionary with results or tuple of (results, artifact) if return_artifact is True
        
    Raises:
        RuntimeError: If version compatibility check fails
    """
    # Check version compatibility
    from .dal_versions import check_compatibility, log_versions, get_version
    compatible, message = check_compatibility('pipeline_v2.1')
    if not compatible:
        log_versions()
        raise RuntimeError(f"Version compatibility error: {message}")
    
    # Log pipeline start with version info
    logger.info("Starting enhanced pipeline with version %s", get_version())
    log_versions()
    
    start_time = time.time()
    
    # Track timing
    vectorization_start = 0
    expansion_start = 0
    
    # Step 1: Initial segmentation
    segments: List[str] = segment_prompt(prompt, n_segments)
    
    # Step 2: Tagging and create SDTs
    tags: List[List[str]] = []
    sdts: List[DALTagBlock] = []
    
    for seg in segments:
        seg_tags = assign_tags(seg)
        tags.append(seg_tags)
        sdts.append(_create_sdt_from_tags(seg_tags))
    
    # Step 3: Vectorization (if requested)
    vectors: Optional[List[List[float]]] = None
    if show_vectors:
        vectorization_start = time.time()
        vectors = [simple_vectorise(seg) for seg in segments]
        vectorization_time = (time.time() - vectorization_start) * 1000  # ms
    
    # Step 4: Optional expansion with ExpanderV2
    expanded_segments: Optional[List[str]] = None
    expansion_time = 0.0
    expansion_metrics = {
        'total_segments': 0,
        'expanded_segments': 0,
        'skipped_short': 0,
        'segment_times': [],
        'input_tokens': [],
        'output_tokens': []
    }
    
    if expand:
        expansion_start = time.time()
        try:
            from .expander_v2_fixed import ExpanderV2
            
            logger.info(f"Initializing ExpanderV2 with model: {expander_model or 'default'}")
            expander = ExpanderV2(expander_model) if expander_model else ExpanderV2()
            expanded_segments = []
            
            for i, seg in enumerate(segments, 1):
                segment_start = time.time()
                expansion_metrics['total_segments'] += 1
                
                try:
                    # Log before expansion
                    input_tokens = len(seg.split())  # Approximate token count
                    expansion_metrics['input_tokens'].append(input_tokens)
                    
                    logger.debug(f"Expanding segment {i}/{len(segments)} ({input_tokens} tokens)")
                    
                    # Perform expansion
                    expanded = expander.expand(seg)
                    
                    # Calculate metrics
                    output_tokens = len(expanded.split())  # Approximate token count
                    segment_time = (time.time() - segment_start) * 1000  # ms
                    
                    expansion_metrics['output_tokens'].append(output_tokens)
                    expansion_metrics['segment_times'].append(segment_time)
                    expansion_metrics['expanded_segments'] += 1
                    
                    logger.debug(f"Expanded segment {i} in {segment_time:.1f}ms "
                               f"({input_tokens}→{output_tokens} tokens, "
                               f"ratio: {output_tokens/max(1, input_tokens):.2f}x)")
                    
                    expanded_segments.append(expanded)
                    
                except Exception as e:
                    logger.warning(f"Error expanding segment {i}: {e}", exc_info=logger.isEnabledFor(logging.DEBUG))
                    expanded_segments.append(seg)  # Fall back to original
                    
        except ImportError as e:
            logger.error("Failed to import ExpanderV2. Ensure all dependencies are installed.", 
                        exc_info=logger.isEnabledFor(logging.DEBUG))
            expanded_segments = None
        
        # Calculate final metrics
        expansion_time = (time.time() - expansion_start) * 1000  # ms
        
        # Log summary if we expanded anything
        if expanded_segments and expansion_metrics['expanded_segments'] > 0:
            avg_time = sum(expansion_metrics['segment_times']) / len(expansion_metrics['segment_times'])
            total_input = sum(expansion_metrics['input_tokens'])
            total_output = sum(expansion_metrics['output_tokens'])
            
            logger.info(
                f"Expanded {expansion_metrics['expanded_segments']}/{expansion_metrics['total_segments']} segments "
                f"in {expansion_time:.1f}ms (avg: {avg_time:.1f}ms/segment, "
                f"tokens: {total_input}→{total_output}, ratio: {total_output/max(1, total_input):.2f}x)"
            )
        elif expanded_segments:
            logger.warning("No segments were expanded. Input may be too short or invalid.")
    
    # Use expanded segments if available, otherwise use original segments
    segments_for_stitch = expanded_segments if expanded_segments else segments
    
    # Step 5: Stitching with Markov chain
    stitching_start = time.time()
    final_answer = stitch_segments(segments_for_stitch, tags)
    stitching_time = (time.time() - stitching_start) * 1000  # ms
    total_time = (time.time() - start_time) * 1000  # ms
    
    # Get version information
    from .dal_versions import get_version, COMPONENT_VERSIONS
    
    # Create and return the final result
    result = {
        'answer': final_answer,
        'segments': segments_for_stitch,
        'tags': tags,
        'vectors': vectors if show_vectors else None,
        'versions': {
            'core': get_version(),
            'components': COMPONENT_VERSIONS,
            'pipeline': 'v2'
        }
    }
    
    if expanded_segments is not None:
        result["expanded_segments"] = expanded_segments
    
    # Create DALRunArtifact if requested
    if return_artifact:
        # Create DALVector
        primary_intent = _extract_primary_intent(prompt)
        dal_vector = DALVector(
            primary_intent=primary_intent,
            sdt=sdts[0] if sdts else DALTagBlock(),
            tokens=[f"token_{i}" for i in range(10)]  # Placeholder
        )
        
        # Create DALBlocks for each segment
        dal_blocks = []
        for i, (seg, tag_list) in enumerate(zip(segments_for_stitch, tags)):
            # Get the vector if available
            vector = vectors[i] if vectors and i < len(vectors) else None
            
            # Get expansion metrics for this segment if available
            segment_metrics = {}
            if expand and i < len(expansion_metrics.get('input_tokens', [])):
                segment_metrics = {
                    'input_tokens': expansion_metrics['input_tokens'][i],
                    'output_tokens': expansion_metrics['output_tokens'][i],
                    'expansion_time_ms': expansion_metrics['segment_times'][i] if i < len(expansion_metrics.get('segment_times', [])) else None
                }
            
            # Create a DALBlock for this segment
            block = DALBlock(
                id=f"segment_{i}",
                text=seg,
                tags=tag_list,
                embedding=vector if vector else None,
                sdt=sdts[i] if i < len(sdts) else DALTagBlock()
            )
            dal_blocks.append(block)
        
        # Create the final artifact
        artifact = DALRunArtifact(
            prompt=prompt,
            dal_vector=dal_vector,
            blocks=dal_blocks,
            answer=final_answer,
            metrics={
                "timing": {
                    "total_ms": total_time,
                    "vectorization_ms": vectorization_time if show_vectors else None,
                    "expansion_ms": expansion_time if expand else None,
                    "stitching_ms": stitching_time,
                    "avg_expansion_ms_per_segment": (
                        sum(expansion_metrics['segment_times']) / len(expansion_metrics['segment_times']) 
                        if expansion_metrics.get('segment_times') else None
                    ) if expand else None
                },
                "tokens": {
                    "raw_prompt_tokens": len(prompt.split()),
                    "compressed_tokens": len(segments),
                    **({
                        "total_input_tokens": sum(expansion_metrics['input_tokens']),
                        "total_output_tokens": sum(expansion_metrics['output_tokens']),
                        "expansion_ratio": sum(expansion_metrics['output_tokens']) / max(1, sum(expansion_metrics['input_tokens'])),
                        "expanded_segments": expansion_metrics['expanded_segments'],
                        "total_segments": expansion_metrics['total_segments']
                    } if expand and expansion_metrics else {})
                }
            }
        )
        
        return (result, artifact) if return_artifact else result
    
    return result

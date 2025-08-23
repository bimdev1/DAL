"""
DAL Pipeline v3 - Enhanced with Local SDT-Aware Generation

This version extends the pipeline with local model generation capabilities
that respect Segment Descriptor Tags (SDTs) for fine-grained control.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import asdict

# Import from v2 to extend functionality
from .pipeline_v2 import (
    _extract_primary_intent,
    _create_sdt_from_tags,
    segment_prompt,
    assign_tags,
    stitch_segments,
    simple_vectorise,
    DALTagBlock,
    DALVector,
    DALBlock,
    DALRunArtifact
)

# Import new components for local generation
from .local_generator import LocalGenerator

logger = logging.getLogger(__name__)

class PipelineConfig:
    """Configuration for the enhanced pipeline."""
    
    def __init__(
        self,
        n_segments: int = 3,
        expand: bool = False,
        show_vectors: bool = False,
        expander_model: Optional[str] = None,
        use_local_generation: bool = False,
        sdt_control: bool = True,
        force_regen: bool = False,
        local_model_spec: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 128,
        local_generation_timeout: float = 8.0
    ):
        """Initialize pipeline configuration.
        
        Args:
            n_segments: Target number of segments for initial segmentation
            expand: Whether to use the expander (legacy)
            show_vectors: Include vector representations in output
            expander_model: Model to use for expansion (legacy)
            use_local_generation: Enable local SDT-aware generation
            sdt_control: Respect SDT tags for generation control
            force_regen: Force regeneration even without SDT tags
            local_model_spec: Configuration for the local model
            max_new_tokens: Maximum tokens to generate per segment
            local_generation_timeout: Timeout in seconds for local generation
        """
        self.n_segments = n_segments
        self.expand = expand
        self.show_vectors = show_vectors
        self.expander_model = expander_model
        self.use_local_generation = use_local_generation
        self.sdt_control = sdt_control
        self.force_regen = force_regen
        self.local_model_spec = local_model_spec or {
            "backend": "huggingface",
            "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "device_map": "auto",
            "load_in_4bit": True,
        }
        self.max_new_tokens = max_new_tokens
        self.local_generation_timeout = local_generation_timeout

def run_enhanced_pipeline_v3(
    prompt: str,
    config: Optional[PipelineConfig] = None,
    return_artifact: bool = False,
) -> Union[Dict[str, object], Tuple[Dict[str, object], DALRunArtifact]]:
    """Run the enhanced DAL pipeline with local generation support.
    
    This version adds support for local SDT-aware generation while maintaining
    backward compatibility with the v2 pipeline.
    
    Args:
        prompt: The natural language prompt to process
        config: Pipeline configuration
        return_artifact: If True, return a DALRunArtifact with rich metadata
        
    Returns:
        Dictionary with results or tuple of (results, artifact) if return_artifact is True
    """
    # Initialize default config if not provided
    if config is None:
        config = PipelineConfig()
    
    # Check version compatibility
    from .dal_versions import check_compatibility, log_versions, get_version
    compatible, message = check_compatibility('pipeline_v3.0')
    if not compatible:
        log_versions()
        raise RuntimeError(f"Version compatibility error: {message}")
    
    # Log pipeline start with version info
    logger.info("Starting enhanced pipeline v3 with version %s", get_version())
    log_versions()
    
    start_time = time.time()
    
    # Initialize local generator if needed
    local_generator = None
    if config.use_local_generation:
        try:
            local_generator = LocalGenerator(config.local_model_spec)
            if not local_generator.is_ready():
                logger.warning("Local generator initialization failed, falling back to standard pipeline")
        except Exception as e:
            logger.error(f"Error initializing local generator: {e}", exc_info=True)
    
    # Track metrics
    metrics = {
        "timing": {"total": 0.0, "segmentation": 0.0, "tagging": 0.0, 
                  "generation": 0.0, "expansion": 0.0, "stitching": 0.0},
        "tokens": {"input": 0, "output": 0, "generated": 0},
        "generation": {
            "segments_processed": 0,
            "segments_regenerated": 0,
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_time": 0.0,
            "per_segment": []
        }
    }
    
    # Step 1: Initial segmentation
    seg_start = time.time()
    segments: List[str] = segment_prompt(prompt, config.n_segments)
    metrics["timing"]["segmentation"] = time.time() - seg_start
    
    # Step 2: Tag segments and create SDTs
    tag_start = time.time()
    tags_list: List[List[str]] = []
    sdts: List[DALTagBlock] = []
    
    for seg in segments:
        seg_tags = assign_tags(seg)
        tags_list.append(seg_tags)
        sdts.append(_create_sdt_from_tags(seg_tags))
    
    metrics["timing"]["tagging"] = time.time() - tag_start
    
    # Step 3: Process each segment with local generation if enabled
    processed_segments = []
    
    gen_start = time.time()
    for i, (seg, sdt) in enumerate(zip(segments, sdts)):
        segment_metrics = {
            "index": i,
            "original_length": len(seg),
            "regenerated": False,
            "tokens_in": 0,
            "tokens_out": 0,
            "time_taken": 0.0,
            "backend": None,
            "model": None,
            "sdt_applied": None
        }
        
        # Check if we should regenerate this segment
        should_regenerate = False
        if local_generator and local_generator.is_ready():
            if config.force_regen:
                should_regenerate = True
                logger.debug(f"Forcing regeneration of segment {i}")
            elif config.sdt_control and _should_regenerate_based_on_sdt(sdt):
                should_regenerate = True
                logger.debug(f"SDT-triggered regeneration for segment {i}")
        
        # Process the segment
        if should_regenerate:
            try:
                gen_seg_start = time.time()
                result = local_generator.generate(
                    text=seg,
                    sdt=sdt,
                    max_new_tokens=config.max_new_tokens,
                    timeout_s=config.local_generation_timeout
                )
                
                # Update metrics
                gen_time = time.time() - gen_seg_start
                processed_segments.append(result["text"])
                
                # Update segment metrics
                segment_metrics.update({
                    "regenerated": True,
                    "tokens_in": result.get("tokens_in", 0),
                    "tokens_out": result.get("tokens_out", 0),
                    "time_taken": gen_time,
                    "backend": result.get("backend"),
                    "model": result.get("model"),
                    "sdt_applied": result.get("sdt_applied"),
                    "prompt_preview": result.get("prompt_preview", "")
                })
                
                # Update global metrics
                metrics["generation"]["segments_regenerated"] += 1
                metrics["generation"]["total_tokens_in"] += segment_metrics["tokens_in"]
                metrics["generation"]["total_tokens_out"] += segment_metrics["tokens_out"]
                metrics["generation"]["total_time"] += gen_time
                
            except Exception as e:
                logger.error(f"Error during local generation for segment {i}: {e}", exc_info=True)
                processed_segments.append(seg)  # Fall back to original
        else:
            # Use original segment
            processed_segments.append(seg)
        
        # Track all processed segments
        metrics["generation"]["per_segment"].append(segment_metrics)
        metrics["generation"]["segments_processed"] += 1
    
    metrics["timing"]["generation"] = time.time() - gen_start
    
    # Step 4: Stitch the final output
    stitch_start = time.time()
    final_text = stitch_segments(processed_segments, tags=tags_list)
    metrics["timing"]["stitching"] = time.time() - stitch_start
    
    # Update total timing
    metrics["timing"]["total"] = time.time() - start_time
    
    # Prepare the result
    result = {
        "text": final_text,
        "segments": processed_segments,
        "tags": tags_list,
        "sdts": [sdt.to_dict() for sdt in sdts],
        "metrics": metrics
    }
    
    # Add vectors if requested
    if config.show_vectors:
        result["vectors"] = [simple_vectorise(seg) for seg in processed_segments]
    
    # Create and return artifact if requested
    if return_artifact:
        # Create the result artifact
        artifact = DALRunArtifact(
            prompt=prompt,
            answer=final_text,
            blocks=[
                DALBlock(
                    id=f"seg-{i:03d}-of-{len(processed_segments):03d}",
                    text=seg, 
                    tags=tags, 
                    vector=simple_vectorise(seg) if config.show_vectors else None,
                    sdt=sdts[i] if i < len(sdts) else DALTagBlock()
                )
                for i, (seg, tags) in enumerate(zip(processed_segments, tags_list))
            ],
            metrics=metrics,
            version=get_version(),
            timestamp=time.time()
        )
        return result, artifact
    
    return result

def _should_regenerate_based_on_sdt(sdt: DALTagBlock) -> bool:
    """Determine if a segment should be regenerated based on its SDT."""
    if not sdt or not hasattr(sdt, 'tags'):
        return False
    
    # Check for explicit regeneration flag
    if hasattr(sdt, 'regenerate') and sdt.regenerate:
        return True
    
    # Check for non-default SDT values that would trigger regeneration
    default_sdt = DALTagBlock()
    for attr in ['tone', 'depth', 'format', 'style']:
        if hasattr(sdt, attr) and getattr(sdt, attr) != getattr(default_sdt, attr, None):
            return True
    
    # Check for any custom instructions
    if hasattr(sdt, 'instructions') and sdt.instructions:
        return True
    
    return False

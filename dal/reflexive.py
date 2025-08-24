"""Reflexive Revision Engine for DAL.

This module provides tools for analyzing and improving generated text segments
through consistency checking, reordering, and stitching.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .vectorizer import simple_vectorise

@dataclass
class ConsistencyIssue:
    """Represents an issue found during consistency checking."""
    block_index: int
    issue_type: str  # 'redundancy' | 'contradiction' | 'tone_mismatch' | 'style_drift'
    severity: float  # 0.0-1.0
    description: str
    related_blocks: List[int] = None

def check_consistency(blocks: List[str]) -> List[ConsistencyIssue]:
    """Check a list of text blocks for consistency issues.
    
    Args:
        blocks: List of text blocks to analyze
        
    Returns:
        List of ConsistencyIssue objects describing any issues found
    """
    issues = []
    
    # Simple redundancy check using vector similarity
    block_vectors = [simple_vectorise(block) for block in blocks]
    
    for i in range(len(blocks)):
        # Check for near-duplicate content
        for j in range(i + 1, len(blocks)):
            # Simple cosine similarity
            sim = np.dot(block_vectors[i], block_vectors[j])
            if sim > 0.9:  # Arbitrary threshold
                issues.append(ConsistencyIssue(
                    block_index=i,
                    issue_type='redundancy',
                    severity=float(sim),
                    description=f"High similarity with block {j}",
                    related_blocks=[j]
                ))
    
    # TODO: Add more sophisticated checks for contradictions, tone, etc.
    # - Contradiction detection
    # - Tone analysis
    # - Style consistency
    
    return issues

def optimize_order(blocks: List[str], tags: List[Any]) -> List[int]:
    """Optimize the order of blocks based on content and SDT tags.
    
    Args:
        blocks: List of text blocks
        tags: List of SDT tag objects or dictionaries for each block
        
    Returns:
        List of indices representing the new order
    """
    # Default order (no reordering)
    order = list(range(len(blocks)))
    if not tags:
        return order
    
    # Handle both dictionary and object access patterns
    def get_tag_depth(tag):
        if tag is None:
            return None
        if hasattr(tag, 'get'):  # Dictionary-like access
            return tag.get('depth')
        if hasattr(tag, 'attributes'):  # DALTagBlock with attributes
            return getattr(tag.attributes, 'depth', None) if hasattr(tag, 'attributes') else None
        return None
    
    # Simple heuristic: if any block has depth=5, try to put summaries first
    depth_5_indices = [i for i, tag in enumerate(tags) 
                      if get_tag_depth(tag) == 5]
    
    if depth_5_indices:
        # Move depth=5 blocks to the front
        remaining = [i for i in order if i not in depth_5_indices]
        order = depth_5_indices + remaining
    
    return order

def reflexive_stitch(
    blocks: List[str], 
    allow_mini_gen: bool = False,
    local_gen: Any = None
) -> Tuple[List[str], Dict[str, Any]]:
    """Improve transitions between blocks and ensure coherence.
    
    Args:
        blocks: List of text blocks to stitch together
        allow_mini_gen: Whether to allow small generations for transitions
        local_gen: Optional LocalGenerator instance for mini-generations
        
    Returns:
        Tuple of (revised_blocks, metadata) where metadata contains
        information about the stitching process
    """
    revised = []
    metadata = {
        'blocks_modified': [],
        'transitions_added': 0,
        'mini_generations': 0
    }
    
    for i, block in enumerate(blocks):
        if i == 0:
            # First block - just add as is
            revised.append(block)
            continue
            
        # Simple transition improvement (can be enhanced)
        prev_block = blocks[i-1]
        
        # Check if we need a transition
        needs_transition = True  # Simple heuristic - could be more sophisticated
        
        if needs_transition and allow_mini_gen and local_gen and local_gen.is_ready():
            # Try to generate a transition
            try:
                transition_prompt = f"Create a transition between these paragraphs:\n\n{prev_block}\n\n[TRANSITION]\n\n{block}"
                result = local_gen.generate(
                    text=transition_prompt,
                    sdt={'tone': 'neutral'},
                    max_new_tokens=50
                )
                if result and result.get('text'):
                    revised[-1] = f"{prev_block}\n\n{result['text'].strip()}"
                    metadata['mini_generations'] += 1
                    metadata['blocks_modified'].append(i-1)
            except Exception as e:
                # Fall through to simple transition
                pass
        
        # Add the current block
        revised.append(block)
    
    return revised, metadata

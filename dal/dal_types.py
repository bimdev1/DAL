"""DAL schema definitions and data structures.

This module defines the core data structures for the Dense Abstract Language (DAL)
framework, including vectors, tags, and export artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime


@dataclass
class DALTagBlock:
    """Structured Synthetic DAL Tags (SDTs) for semantic positioning and control."""
    version: str = "0.1.0"
    topic: str = "unknown"
    voice: str = "neutral"
    depth: int = 3
    format: str = "exposition"
    priority: int = 0
    recursion_limit: int = 1
    source: str = "inherited"  # Tracks tag provenance for reflexive passes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the DALTagBlock to a dictionary."""
        return {
            "version": self.version,
            "topic": self.topic,
            "voice": self.voice,
            "depth": self.depth,
            "format": self.format,
            "priority": self.priority,
            "recursion_limit": self.recursion_limit,
            "source": self.source
        }


@dataclass
class GuidanceMeta:
    """Metadata for guiding generation and reflexive passes."""
    goal: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class DALVector:
    """Compact representation of a prompt's semantic intent and structure."""
    version: str = "0.1.0"
    primary_intent: str = ""
    tokens: List[str] = field(default_factory=list)  # Placeholder for pseudo-tokens
    sdt: DALTagBlock = field(default_factory=DALTagBlock)
    guidance: GuidanceMeta = field(default_factory=GuidanceMeta)


@dataclass
class DALBlock:
    """A segment or sub-block of generated content with metadata."""
    id: str  # Format: "seg-{idx:03d}-of-{total:03d}"
    text: str
    tags: List[str]
    vector: Optional[List[float]] = None  # Optional vector representation of the text
    embedding: Optional[List[float]] = None
    sdt: DALTagBlock = field(default_factory=DALTagBlock)


@dataclass
class DALRunArtifact:
    """Top-level container for DAL pipeline outputs and metadata."""
    version: Dict[str, str] = field(
        default_factory=lambda: {
            "dal_export": "0.1.0",
            "dal_vector": "0.1.0",
            "sdt": "0.1.0"
        }
    )
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    prompt: str = ""
    dal_vector: DALVector = field(default_factory=DALVector)
    blocks: List[DALBlock] = field(default_factory=list)
    answer: str = ""
    metrics: Dict[str, Any] = field(
        default_factory=lambda: {
            "timing": {
                "vectorization_ms": None,
                "expansion_ms": None,
                "stitching_ms": None
            },
            "tokens": {
                "raw_prompt_tokens": None,
                "compressed_tokens": None
            }
        }
    )

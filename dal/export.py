"""DAL export utilities for serializing pipeline artifacts."""

import json
from typing import Any, Dict, Optional
from pathlib import Path

from .dal_types import DALRunArtifact


def to_dict(artifact: DALRunArtifact) -> Dict[str, Any]:
    """Convert a DALRunArtifact to a dictionary for JSON serialization.
    
    Args:
        artifact: The DALRunArtifact to serialize.
        
    Returns:
        A dictionary ready for JSON serialization.
    """
    # Use the dataclasses.asdict() approach for nested dataclasses
    def _to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {k: _to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, (list, tuple)):
            return [_to_dict(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        else:
            return obj
    
    return _to_dict(artifact)


def save_json(artifact: DALRunArtifact, path: str) -> None:
    """Save a DALRunArtifact to a JSON file.
    
    Args:
        artifact: The DALRunArtifact to save.
        path: The file path where the JSON will be written.
        
    Raises:
        IOError: If the file cannot be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(to_dict(artifact), f, indent=2, ensure_ascii=False)


def to_tso(artifact: DALRunArtifact) -> Dict[str, Any]:
    """Convert a DALRunArtifact to a Tagged Semantic Object (TSO) format.
    
    This is a stable-key variant designed for inter-system exchange.
    
    Args:
        artifact: The DALRunArtifact to convert.
        
    Returns:
        A dictionary with stable keys for TSO serialization.
    """
    # For now, just return the standard dict representation
    # This can be specialized later if needed
    return to_dict(artifact)

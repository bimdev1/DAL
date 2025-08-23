"""Top level package for the simplified DAL implementation.

This module exposes the high‑level ``run_pipeline`` function for
turning a prompt into a stitched answer as well as lower level
components for custom workflows.  See the README in the repository
root for usage instructions.
"""

from .vectorizer import simple_vectorise
from .tagging import assign_tags
from .segmenter import segment_prompt
from .markov import MarkovChain
from .stitcher import stitch_segments
from .pipeline import run_pipeline
from .pipeline_v2 import run_enhanced_pipeline
from .pipeline_v3 import run_enhanced_pipeline_v3, PipelineConfig
from .expander_v2_fixed import ExpanderV2
from .local_generator import LocalGenerator
from .sdt_injector import build_prompt, SDTApplied
from .dal_versions import (
    get_version,
    get_version_info,
    check_compatibility,
    log_versions,
    __version__,
    __version_info__
)

# DAL Schema and Export
from .dal_types import DALTagBlock, GuidanceMeta, DALVector, DALBlock, DALRunArtifact
from .export import to_dict, save_json, to_tso

__all__ = [
    # Core pipeline
    "simple_vectorise",
    "assign_tags",
    "segment_prompt",
    "MarkovChain",
    "stitch_segments",
    "run_pipeline",
    "run_enhanced_pipeline",
    "run_enhanced_pipeline_v3",
    "PipelineConfig",
    "LocalGenerator",
    "build_prompt",
    "SDTApplied",
    "ExpanderV2",
    
    # Version information
    "get_version",
    "get_version_info",
    "check_compatibility",
    "log_versions",
    "__version__",
    "__version_info__",
    
    # DAL Schema and Export
    "DALTagBlock",
    "GuidanceMeta",
    "DALVector",
    "DALBlock",
    "DALRunArtifact",
    "to_dict",
    "save_json",
    "to_tso",
]
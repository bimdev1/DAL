"""
DAL Version Management

This module provides version information and compatibility checks for the DAL system.
It follows semantic versioning (MAJOR.MINOR.PATCH) and includes functionality
to check version compatibility and log version information.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Version information for the DAL system
# Format: (MAJOR, MINOR, PATCH, STATUS)
VERSION = (1, 2, 0, 'beta')

# Version history with changes
VERSION_HISTORY = {
    '1.2.0': 'Added local SDT-aware generation with model_loader and pipeline_v3',
    '1.1.0': 'Added version management system and improved pipeline metrics',
    '1.0.0': 'Initial release with ExpanderV2 improvements and version tracking',
}

# Component versions
COMPONENT_VERSIONS = {
    'core': '1.2.0',
    'expander': '1.1.0',  # ExpanderV2
    'pipeline': '1.2.0',  # pipeline_v3
    'pipeline_v2': '1.1.0',
    'pipeline_v3': '1.0.0',
    'model_loader': '1.0.0',
    'local_generator': '1.0.0',
    'sdt_injector': '1.0.0',
    'vectorizer': '1.0.0',
    'tagging': '1.0.0',
}

# Version compatibility matrix
# Defines minimum required versions for compatibility
COMPATIBILITY = {
    'pipeline_v1': {'min_core': '0.9.0', 'max_core': '1.0.0'},
    'pipeline_v2': {'min_core': '1.0.0', 'max_core': '1.2.0'},
    'pipeline_v2.1': {'min_core': '1.1.0', 'max_core': '1.2.0'},
    'pipeline_v3': {'min_core': '1.2.0', 'max_core': '2.0.0'},
    'expander_v1': {'min_core': '0.9.0', 'max_core': '1.0.0'},
    'expander_v2': {'min_core': '1.0.0', 'max_core': '2.0.0'},
    'model_loader': {'min_core': '1.2.0', 'max_core': '2.0.0'},
    'local_generator': {'min_core': '1.2.0', 'max_core': '2.0.0'},
    'sdt_injector': {'min_core': '1.2.0', 'max_core': '2.0.0'},
}


@dataclass
class VersionInfo:
    """Structured version information."""
    major: int
    minor: int
    patch: int
    status: str = 'final'
    
    def __str__(self) -> str:
        """Return version as string (e.g., '1.0.0' or '1.0.0-beta')."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.status != 'final':
            version += f"-{self.status}"
        return version
    
    @classmethod
    def from_string(cls, version_str: str) -> 'VersionInfo':
        """Create VersionInfo from version string."""
        if '-' in version_str:
            version_part, status = version_str.split('-', 1)
        else:
            version_part, status = version_str, 'final'
            
        major, minor, patch = map(int, version_part.split('.'))
        return cls(major, minor, patch, status)
    
    def __lt__(self, other: 'VersionInfo') -> bool:
        """Compare versions for less than."""
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch
        # Consider status for pre-releases (e.g., 1.0.0-beta < 1.0.0)
        if self.status != other.status:
            if self.status == 'final':
                return False
            if other.status == 'final':
                return True
            return self.status < other.status
        return False
    
    def __eq__(self, other: object) -> bool:
        """Check if versions are equal."""
        if not isinstance(other, VersionInfo):
            return False
        return (self.major == other.major and 
                self.minor == other.minor and 
                self.patch == other.patch and
                self.status == other.status)
    
    def compatible_with(self, other: 'VersionInfo') -> bool:
        """Check if this version is compatible with another version."""
        # Major versions must match for compatibility
        return self.major == other.major


def get_version() -> str:
    """Get the current DAL version as a string."""
    version = f"{VERSION[0]}.{VERSION[1]}.{VERSION[2]}"
    if VERSION[3] != 'final':
        version += f"-{VERSION[3]}"
    return version


def get_version_info() -> VersionInfo:
    """Get the current DAL version as a VersionInfo object."""
    return VersionInfo(*VERSION)


def check_compatibility(
    component: str,
    min_version: Optional[str] = None,
    max_version: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Check if the current version is compatible with the specified requirements.
    
    Args:
        component: The component name to check compatibility for
        min_version: Minimum required version (inclusive)
        max_version: Maximum allowed version (exclusive)
        
    Returns:
        Tuple of (is_compatible, message)
    """
    current_version = get_version_info()
    
    if min_version:
        min_ver = VersionInfo.from_string(min_version)
        if current_version < min_ver:
            return False, f"Version {current_version} is older than minimum required {min_version}"
    
    if max_version:
        max_ver = VersionInfo.from_string(max_version)
        if not (current_version < max_ver):
            return False, f"Version {current_version} is not older than maximum allowed {max_version}"
    
    # Check component-specific compatibility
    if component in COMPATIBILITY:
        reqs = COMPATIBILITY[component]
        if 'min_core' in reqs:
            min_core = VersionInfo.from_string(reqs['min_core'])
            if current_version < min_core:
                return False, f"Core version {current_version} is older than minimum required {reqs['min_core']} for {component}"
        
        if 'max_core' in reqs:
            max_core = VersionInfo.from_string(reqs['max_core'])
            if not (current_version < max_core):
                return False, f"Core version {current_version} is not older than maximum allowed {reqs['max_core']} for {component}"
    
    return True, f"Version {current_version} is compatible with {component}"


def log_versions() -> None:
    """Log version information for all components."""
    logger.info("DAL System Versions:")
    logger.info(f"  Core: {get_version()}")
    for component, version in sorted(COMPONENT_VERSIONS.items()):
        logger.info(f"  {component.capitalize()}: {version}")
    
    # Log compatibility information
    logger.debug("Compatibility Matrix:")
    for component, reqs in COMPATIBILITY.items():
        logger.debug(f"  {component}: {reqs}")


# Initialize logging of version information on import
log_versions()

# Add version to module namespace
__version__ = get_version()
__version_info__ = get_version_info()

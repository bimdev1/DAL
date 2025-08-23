#!/usr/bin/env python3
"""
Test script for DAL versioning functionality.
"""

import sys
import logging
from dal import (
    get_version,
    get_version_info,
    check_compatibility,
    log_versions,
    __version__,
    __version_info__
)

def test_version_basics():
    """Test basic version functionality."""
    print("\n=== Testing Version Basics ===")
    
    # Test get_version()
    version_str = get_version()
    print(f"Version string: {version_str}")
    assert isinstance(version_str, str), "Version should be a string"
    assert len(version_str.split('.')) >= 2, "Version should have at least MAJOR.MINOR"
    
    # Test get_version_info()
    version_info = get_version_info()
    print(f"Version info: {version_info}")
    assert version_info.major >= 0, "Major version should be non-negative"
    assert version_info.minor >= 0, "Minor version should be non-negative"
    assert version_info.patch >= 0, "Patch version should be non-negative"
    
    # Test __version__ and __version_info__
    print(f"Module __version__: {__version__}")
    print(f"Module __version_info__: {__version_info__}")
    assert __version__ == version_str, "Module __version__ should match get_version()"
    assert __version_info__ == version_info, "Module __version_info__ should match get_version_info()"
    
    print("✓ Version basics test passed!")

def test_version_comparison():
    """Test version comparison functionality."""
    print("\n=== Testing Version Comparison ===")
    
    from dal.dal_versions import VersionInfo
    
    # Test basic comparisons
    v1 = VersionInfo(1, 0, 0)
    v1_1 = VersionInfo(1, 1, 0)
    v2 = VersionInfo(2, 0, 0)
    v1_0_1 = VersionInfo(1, 0, 1)
    
    assert v1 < v1_1, "1.0.0 should be less than 1.1.0"
    assert v1_1 < v2, "1.1.0 should be less than 2.0.0"
    assert v1 < v1_0_1, "1.0.0 should be less than 1.0.1"
    assert v1 == VersionInfo(1, 0, 0), "1.0.0 should equal 1.0.0"
    
    # Test string parsing
    assert VersionInfo.from_string("1.2.3") == VersionInfo(1, 2, 3)
    assert VersionInfo.from_string("1.2.3-beta") == VersionInfo(1, 2, 3, "beta")
    
    print("✓ Version comparison test passed!")

def test_compatibility():
    """Test version compatibility checks."""
    print("\n=== Testing Version Compatibility ===")
    
    # Test basic compatibility
    compatible, msg = check_compatibility("pipeline_v2")
    print(f"Pipeline v2 compatibility: {msg}")
    assert compatible, "Current version should be compatible with pipeline_v2"
    
    # Test with version constraints
    compatible, msg = check_compatibility("pipeline_v2", min_version="0.9.0")
    print(f"Pipeline v2 with min_version=0.9.0: {msg}")
    assert compatible, "Should be compatible with min_version=0.9.0"
    
    # Test with invalid version
    compatible, msg = check_compatibility("pipeline_v2", min_version="99.0.0")
    print(f"Pipeline v2 with min_version=99.0.0: {msg}")
    assert not compatible, "Should not be compatible with min_version=99.0.0"
    
    print("✓ Version compatibility test passed!")

def test_log_versions():
    """Test the log_versions function."""
    print("\n=== Testing Version Logging ===")
    
    # Capture log output
    import io
    from contextlib import redirect_stdout
    
    stream = io.StringIO()
    with redirect_stdout(stream):
        # Configure logging to output to our stream
        logger = logging.getLogger('dal.dal_versions')
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Call the function
        log_versions()
        
        # Remove the handler
        logger.removeHandler(handler)
    
    # Check the output
    output = stream.getvalue()
    print(f"Log output:\n{output}")
    assert "DAL System Versions:" in output, "Should log system versions"
    assert "Core:" in output, "Should log core version"
    
    print("✓ Version logging test passed!")

def test_pipeline_versioning():
    """Test that the pipeline includes version information."""
    print("\n=== Testing Pipeline Versioning ===")
    
    from dal.pipeline_v2 import run_enhanced_pipeline
    
    # Run a simple pipeline
    result, artifact = run_enhanced_pipeline(
        "Test prompt for versioning",
        return_artifact=True
    )
    
    # Check that versions are included in the result
    assert 'versions' in result, "Result should include versions"
    assert 'core' in result['versions'], "Versions should include core version"
    assert 'components' in result['versions'], "Versions should include components"
    
    # Check that version information is included in the artifact
    assert hasattr(artifact, 'version'), "Artifact should have version information"
    assert isinstance(artifact.version, dict), "Version should be a dictionary"
    assert 'dal_export' in artifact.version, "Version should include dal_export"
    assert hasattr(artifact, 'dal_vector'), "Artifact should have a DALVector"
    assert hasattr(artifact.dal_vector, 'version'), "DALVector should have version information"
    
    print("✓ Pipeline versioning test passed!")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    test_version_basics()
    test_version_comparison()
    test_compatibility()
    test_log_versions()
    test_pipeline_versioning()
    
    print("\nAll versioning tests passed successfully!")

#!/usr/bin/env python3
"""
Integration test for DAL versioning functionality.

This script demonstrates how to use the versioning features in the DAL system,
including checking compatibility and accessing version information.
"""

import logging
from dal.dal_versions import (
    get_version,
    get_version_info,
    check_compatibility,
    log_versions,
    COMPONENT_VERSIONS
)
from dal.pipeline_v2 import run_enhanced_pipeline

def test_version_integration():
    """Test the integration of versioning in the DAL system."""
    print("\n=== Testing DAL Version Integration ===")
    
    # 1. Basic version information
    print("\n1. Basic Version Information:")
    print(f"DAL Version: {get_version()}")
    version_info = get_version_info()
    print(f"Version Info: {version_info}")
    
    # 2. Component versions
    print("\n2. Component Versions:")
    for component, version in COMPONENT_VERSIONS.items():
        print(f"  - {component}: {version}")
    
    # 3. Compatibility checks
    print("\n3. Compatibility Checks:")
    components = [
        'pipeline_v2.1',
        'expander_v2',
        'pipeline_v1',  # Should be incompatible with current version
    ]
    
    for component in components:
        compatible, message = check_compatibility(component)
        status = "✓" if compatible else "✗"
        print(f"  - {status} {component}: {message}")
    
    # 4. Pipeline integration
    print("\n4. Pipeline Version Integration:")
    try:
        result, artifact = run_enhanced_pipeline(
            "Demonstrate version integration in the pipeline",
            return_artifact=True
        )
        
        print("Pipeline executed successfully with versions:")
        print(f"  - Core: {result['versions']['core']}")
        print(f"  - Pipeline: {result['versions']['pipeline']}")
        print(f"  - Artifact Version: {artifact.version}")
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        raise
    
    print("\n✓ Version integration test completed successfully!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the test
    test_version_integration()

#!/usr/bin/env python3
"""
Simple script to check DAL version information.
"""

import logging
from dal.dal_versions import (
    get_version,
    get_version_info,
    check_compatibility,
    log_versions,
    COMPONENT_VERSIONS
)

def main():
    print("\n=== DAL Version Information ===\n")
    
    # Basic version info
    print(f"DAL Version: {get_version()}")
    print(f"Version Info: {get_version_info()}")
    
    # Component versions
    print("\nComponent Versions:")
    for component, version in COMPONENT_VERSIONS.items():
        print(f"  - {component}: {version}")
    
    # Compatibility checks
    print("\nCompatibility Checks:")
    components = [
        'pipeline_v2.1',
        'expander_v2',
        'pipeline_v1',
    ]
    
    for component in components:
        compatible, message = check_compatibility(component)
        status = "✓" if compatible else "✗"
        print(f"  - {status} {component}: {message}")
    
    # Log versions
    print("\nLog output:")
    log_versions()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

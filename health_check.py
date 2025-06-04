#!/usr/bin/env python3
"""
Health Check Script

Standalone script for checking system health.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.health_monitoring import initialize_health_monitor
import json

def main():
    project_root = os.path.dirname(__file__)
    monitor = initialize_health_monitor(project_root)
    
    health_status = monitor.check_system_health()
    
    print(json.dumps(health_status, indent=2))
    
    # Exit with error code if not healthy
    if health_status['overall_status'] != 'healthy':
        sys.exit(1)

if __name__ == "__main__":
    main()

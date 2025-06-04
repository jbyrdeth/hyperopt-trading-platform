#!/usr/bin/env python3
"""
API Server Startup Script

Run this script to start the Trading Strategy Optimization API server.
This avoids conflicts with the root main.py file.
"""

import uvicorn
import sys
import os

if __name__ == "__main__":
    # Ensure we're running from the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("🚀 Starting Trading Strategy Optimization API Server...")
    print(f"📂 Working directory: {os.getcwd()}")
    print("📊 Dashboard will be available at: http://localhost:8000/api/v1/monitoring/")
    print("📋 API docs will be available at: http://localhost:8000/api/docs")
    print("📈 Metrics endpoint: http://localhost:8000/metrics")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 
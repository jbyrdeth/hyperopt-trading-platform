#!/usr/bin/env python3
"""
Export Functionality Test
Tests Pine Script and PDF export capabilities using the correct API structure.
"""

import requests
import json
from datetime import datetime

# Configuration
API_BASE = "http://localhost:8000/api/v1"
API_KEY = "dev_key_123"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

def test_pine_script_export():
    """Test Pine Script export with optimization results."""
    print("🌲 Testing Pine Script Export")
    
    # Sample optimization results
    optimization_results = {
        "strategy_name": "MovingAverageCrossover",
        "best_parameters": {
            "fast_period": 12,
            "slow_period": 26,
            "signal_threshold": 0.02
        },
        "performance_metrics": {
            "total_return": 45.2,
            "sharpe_ratio": 1.85,
            "max_drawdown": 12.5,
            "win_rate": 0.68,
            "trades_count": 156
        }
    }
    
    pine_request = {
        "strategy_name": "MovingAverageCrossover",
        "optimization_results": optimization_results,
        "output_format": "strategy",  # or "indicator"
        "include_validation": True,
        "include_comments": True
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/export/pine-script",
            headers=HEADERS,
            json=pine_request,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Pine Script generated successfully")
            print(f"   File ID: {data.get('file_id')}")
            print(f"   File size: {data.get('file_size')} bytes")
            print(f"   Download URL: {data.get('download_url')}")
            
            # Test preview
            preview = data.get('script_preview', '')
            if len(preview) > 50:
                print(f"   Preview: {preview[:100]}...")
            
            return data.get('file_id')
        else:
            print(f"❌ Pine Script export failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Pine Script export error: {e}")
        return None

def test_pdf_report_export():
    """Test PDF report export with optimization results."""
    print("\n📄 Testing PDF Report Export")
    
    # Sample optimization results
    optimization_results = {
        "strategy_name": "MovingAverageCrossover",
        "best_parameters": {
            "fast_period": 12,
            "slow_period": 26,
            "signal_threshold": 0.02
        },
        "performance_metrics": {
            "total_return": 45.2,
            "sharpe_ratio": 1.85,
            "sortino_ratio": 2.1,
            "calmar_ratio": 1.3,
            "max_drawdown": 12.5,
            "volatility": 18.7,
            "win_rate": 0.68,
            "profit_factor": 1.75,
            "trades_count": 156
        },
        "validation_results": {
            "out_of_sample_return": 38.5,
            "cross_validation_score": 0.82
        }
    }
    
    report_request = {
        "strategy_name": "MovingAverageCrossover",
        "optimization_results": optimization_results,
        "report_type": "full",
        "include_charts": True,
        "include_detailed_tables": True,
        "template": "professional"
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/export/report",
            headers=HEADERS,
            json=report_request,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ PDF Report generated successfully")
            print(f"   File ID: {data.get('file_id')}")
            print(f"   File size: {data.get('file_size')} bytes")
            print(f"   Pages: {data.get('page_count')}")
            print(f"   Download URL: {data.get('download_url')}")
            
            return data.get('file_id')
        else:
            print(f"❌ PDF Report export failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ PDF Report export error: {e}")
        return None

def test_file_download(file_id: str, file_type: str):
    """Test file download functionality."""
    print(f"\n💾 Testing {file_type} File Download")
    
    try:
        response = requests.get(
            f"{API_BASE}/export/download/{file_id}",
            headers=HEADERS,
            timeout=30
        )
        
        if response.status_code == 200:
            # Determine file extension
            extension = ".pine" if file_type == "Pine Script" else ".pdf"
            filename = f"test_download_{file_id}{extension}"
            
            # Save file
            with open(filename, "wb") as f:
                f.write(response.content)
            
            print(f"✅ {file_type} downloaded successfully")
            print(f"   File saved: {filename}")
            print(f"   File size: {len(response.content)} bytes")
            
            return True
        else:
            print(f"❌ {file_type} download failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ {file_type} download error: {e}")
        return False

def test_file_listing():
    """Test file listing functionality."""
    print("\n📋 Testing File Listing")
    
    try:
        response = requests.get(
            f"{API_BASE}/export/files",
            headers=HEADERS,
            timeout=10
        )
        
        if response.status_code == 200:
            files = response.json()
            print(f"✅ File listing successful")
            print(f"   Available files: {len(files)}")
            
            for file_info in files[:3]:  # Show first 3 files
                print(f"   • {file_info.get('filename')} ({file_info.get('file_size')} bytes)")
            
            return True
        else:
            print(f"❌ File listing failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ File listing error: {e}")
        return False

def main():
    """Run export functionality tests."""
    print("🧪 EXPORT FUNCTIONALITY TEST")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests_passed = 0
    total_tests = 0
    
    # Test Pine Script Export
    total_tests += 1
    pine_file_id = test_pine_script_export()
    if pine_file_id:
        tests_passed += 1
        
        # Test Pine Script download
        total_tests += 1
        if test_file_download(pine_file_id, "Pine Script"):
            tests_passed += 1
    
    # Test PDF Report Export
    total_tests += 1
    pdf_file_id = test_pdf_report_export()
    if pdf_file_id:
        tests_passed += 1
        
        # Test PDF download
        total_tests += 1
        if test_file_download(pdf_file_id, "PDF Report"):
            tests_passed += 1
    
    # Test file listing
    total_tests += 1
    if test_file_listing():
        tests_passed += 1
    
    # Results
    print("\n" + "=" * 50)
    print("🏁 EXPORT TESTS COMPLETE")
    print(f"Tests Passed: {tests_passed}/{total_tests} ({tests_passed/total_tests*100:.1f}%)")
    
    if tests_passed == total_tests:
        print("🎉 ALL EXPORT TESTS PASSED!")
        return True
    else:
        print(f"❌ {total_tests - tests_passed} EXPORT TESTS FAILED")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 
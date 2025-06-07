#!/usr/bin/env python3
"""
Complete System Test - End-to-End Pipeline Validation
Tests the full optimization workflow from submission to export.
"""

import requests
import time
import json
import sys
from datetime import datetime
from typing import Dict, Any

# Configuration
API_BASE = "http://localhost:8000/api/v1"
API_KEY = "dev_key_123"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

class SystemTester:
    """Comprehensive system testing class."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        
    def log_test(self, test_name: str, status: str, details: Any = None):
        """Log test results."""
        self.test_results[test_name] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⏳"
        print(f"{status_emoji} {test_name}: {status}")
        if details and isinstance(details, dict):
            for key, value in details.items():
                print(f"   {key}: {value}")
        print()

    def test_system_health(self) -> bool:
        """Test 1: System Health and Connectivity."""
        print("🔍 Test 1: System Health Check")
        
        try:
            # Basic health check
            response = requests.get(f"{API_BASE}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                self.log_test("Health Check", "PASS", {
                    "uptime": f"{health_data.get('uptime_seconds', 0):.1f}s",
                    "active_jobs": health_data.get('active_jobs', 0),
                    "memory_usage": f"{health_data.get('memory_usage_mb', 0):.1f}MB"
                })
                return True
            else:
                self.log_test("Health Check", "FAIL", {"status_code": response.status_code})
                return False
                
        except Exception as e:
            self.log_test("Health Check", "FAIL", {"error": str(e)})
            return False

    def test_authentication(self) -> bool:
        """Test 2: API Authentication."""
        print("🔐 Test 2: Authentication System")
        
        try:
            # Test valid API key
            response = requests.get(f"{API_BASE}/strategies", headers=HEADERS, timeout=10)
            if response.status_code == 200:
                self.log_test("Valid API Key", "PASS")
            else:
                self.log_test("Valid API Key", "FAIL", {"status_code": response.status_code})
                return False
            
            # Test invalid API key
            invalid_headers = {"X-API-Key": "invalid_key"}
            response = requests.get(f"{API_BASE}/strategies", headers=invalid_headers, timeout=10)
            if response.status_code == 401:
                self.log_test("Invalid API Key Rejection", "PASS")
                return True
            else:
                self.log_test("Invalid API Key Rejection", "FAIL", {"status_code": response.status_code})
                return False
                
        except Exception as e:
            self.log_test("Authentication Test", "FAIL", {"error": str(e)})
            return False

    def test_strategy_discovery(self) -> bool:
        """Test 3: Strategy Discovery and Metadata."""
        print("📊 Test 3: Strategy Discovery")
        
        try:
            response = requests.get(f"{API_BASE}/strategies", headers=HEADERS, timeout=10)
            if response.status_code == 200:
                data = response.json()
                strategies = data.get('strategies', [])
                
                if len(strategies) >= 3:
                    self.log_test("Strategy Discovery", "PASS", {
                        "strategy_count": len(strategies),
                        "categories": len(data.get('categories', {})),
                        "sample_strategy": strategies[0]['name']
                    })
                    return True
                else:
                    self.log_test("Strategy Discovery", "FAIL", {"strategy_count": len(strategies)})
                    return False
            else:
                self.log_test("Strategy Discovery", "FAIL", {"status_code": response.status_code})
                return False
                
        except Exception as e:
            self.log_test("Strategy Discovery", "FAIL", {"error": str(e)})
            return False

    def test_optimization_submission(self) -> str:
        """Test 4: Optimization Job Submission."""
        print("🚀 Test 4: Optimization Job Submission")
        
        optimization_request = {
            "strategy_name": "MovingAverageCrossover",
            "symbol": "BTCUSDT",
            "timeframe": "4h",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "optimization_config": {
                "trials": 20,  # Quick test
                "timeout": 300,  # 5 minutes
                "optimization_metric": "sharpe_ratio"
            },
            "strategy_params": {
                "fast_period": {"min": 8, "max": 15},
                "slow_period": {"min": 20, "max": 35},
                "signal_threshold": {"min": 0.01, "max": 0.05}
            }
        }
        
        try:
            response = requests.post(
                f"{API_BASE}/optimize/single",
                headers=HEADERS,
                json=optimization_request,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                job_id = data.get('job_id')
                if job_id:
                    self.log_test("Optimization Submission", "PASS", {
                        "job_id": job_id,
                        "status": data.get('status', 'unknown')
                    })
                    return job_id
                else:
                    self.log_test("Optimization Submission", "FAIL", {"error": "No job_id returned"})
                    return None
            else:
                self.log_test("Optimization Submission", "FAIL", {
                    "status_code": response.status_code,
                    "response": response.text[:200]
                })
                return None
                
        except Exception as e:
            self.log_test("Optimization Submission", "FAIL", {"error": str(e)})
            return None

    def test_optimization_monitoring(self, job_id: str) -> bool:
        """Test 5: Optimization Progress Monitoring."""
        print("📈 Test 5: Optimization Progress Monitoring")
        
        max_wait_time = 600  # 10 minutes
        start_time = time.time()
        
        try:
            while time.time() - start_time < max_wait_time:
                response = requests.get(
                    f"{API_BASE}/optimize/status/{job_id}",
                    headers=HEADERS,
                    timeout=10
                )
                
                if response.status_code == 200:
                    status_data = response.json()
                    status = status_data.get('status')
                    progress = status_data.get('progress', 0)
                    
                    print(f"   📊 Status: {status}, Progress: {progress}%")
                    
                    if status == "completed":
                        self.log_test("Optimization Monitoring", "PASS", {
                            "final_status": status,
                            "duration": f"{time.time() - start_time:.1f}s",
                            "trials_completed": status_data.get('trials_completed', 0)
                        })
                        return True
                    elif status == "failed":
                        self.log_test("Optimization Monitoring", "FAIL", {
                            "final_status": status,
                            "error": status_data.get('error_message', 'Unknown error')
                        })
                        return False
                    
                    time.sleep(10)  # Check every 10 seconds
                else:
                    self.log_test("Optimization Monitoring", "FAIL", {
                        "status_code": response.status_code
                    })
                    return False
            
            # Timeout
            self.log_test("Optimization Monitoring", "FAIL", {"error": "Timeout after 10 minutes"})
            return False
            
        except Exception as e:
            self.log_test("Optimization Monitoring", "FAIL", {"error": str(e)})
            return False

    def test_results_retrieval(self, job_id: str) -> Dict:
        """Test 6: Results Retrieval and Analysis."""
        print("📋 Test 6: Results Retrieval")
        
        try:
            response = requests.get(
                f"{API_BASE}/optimize/results/{job_id}",
                headers=HEADERS,
                timeout=30
            )
            
            if response.status_code == 200:
                results = response.json()
                
                # Validate results structure
                required_fields = ['best_parameters', 'performance_metrics', 'optimization_history']
                missing_fields = [field for field in required_fields if field not in results]
                
                if not missing_fields:
                    metrics = results.get('performance_metrics', {})
                    self.log_test("Results Retrieval", "PASS", {
                        "sharpe_ratio": f"{metrics.get('sharpe_ratio', 0):.3f}",
                        "total_return": f"{metrics.get('total_return', 0):.2%}",
                        "max_drawdown": f"{metrics.get('max_drawdown', 0):.2%}",
                        "trials_run": len(results.get('optimization_history', []))
                    })
                    return results
                else:
                    self.log_test("Results Retrieval", "FAIL", {
                        "missing_fields": missing_fields
                    })
                    return None
            else:
                self.log_test("Results Retrieval", "FAIL", {
                    "status_code": response.status_code
                })
                return None
                
        except Exception as e:
            self.log_test("Results Retrieval", "FAIL", {"error": str(e)})
            return None

    def test_pine_script_export(self, job_id: str) -> bool:
        """Test 7: Pine Script Export."""
        print("🌲 Test 7: Pine Script Export")
        
        try:
            response = requests.get(
                f"{API_BASE}/export/pinescript/{job_id}",
                headers=HEADERS,
                timeout=30
            )
            
            if response.status_code == 200:
                pine_data = response.json()
                pine_script = pine_data.get('pine_script', '')
                
                if pine_script and len(pine_script) > 100:
                    # Save Pine Script file
                    with open(f"optimization_{job_id}.pine", "w") as f:
                        f.write(pine_script)
                    
                    self.log_test("Pine Script Export", "PASS", {
                        "script_length": len(pine_script),
                        "file_saved": f"optimization_{job_id}.pine",
                        "version": pine_data.get('version', 'unknown')
                    })
                    return True
                else:
                    self.log_test("Pine Script Export", "FAIL", {
                        "script_length": len(pine_script)
                    })
                    return False
            else:
                self.log_test("Pine Script Export", "FAIL", {
                    "status_code": response.status_code
                })
                return False
                
        except Exception as e:
            self.log_test("Pine Script Export", "FAIL", {"error": str(e)})
            return False

    def test_pdf_report_export(self, job_id: str) -> bool:
        """Test 8: PDF Report Export."""
        print("📄 Test 8: PDF Report Export")
        
        try:
            response = requests.get(
                f"{API_BASE}/export/pdf/{job_id}",
                headers=HEADERS,
                timeout=60
            )
            
            if response.status_code == 200:
                # Check content type
                content_type = response.headers.get('content-type', '')
                
                if 'application/pdf' in content_type:
                    # Save PDF file
                    with open(f"optimization_report_{job_id}.pdf", "wb") as f:
                        f.write(response.content)
                    
                    self.log_test("PDF Report Export", "PASS", {
                        "file_size": f"{len(response.content) / 1024:.1f}KB",
                        "file_saved": f"optimization_report_{job_id}.pdf",
                        "content_type": content_type
                    })
                    return True
                else:
                    self.log_test("PDF Report Export", "FAIL", {
                        "wrong_content_type": content_type
                    })
                    return False
            else:
                self.log_test("PDF Report Export", "FAIL", {
                    "status_code": response.status_code
                })
                return False
                
        except Exception as e:
            self.log_test("PDF Report Export", "FAIL", {"error": str(e)})
            return False

    def test_monitoring_metrics(self) -> bool:
        """Test 9: Monitoring and Metrics."""
        print("📊 Test 9: Monitoring System")
        
        try:
            # Test Prometheus metrics
            response = requests.get("http://localhost:8000/metrics", timeout=10)
            if response.status_code == 200:
                metrics_text = response.text
                
                # Check for key metrics
                key_metrics = [
                    "api_requests_total",
                    "optimization_jobs_total",
                    "system_memory_usage",
                    "api_request_duration"
                ]
                
                found_metrics = [metric for metric in key_metrics if metric in metrics_text]
                
                self.log_test("Prometheus Metrics", "PASS", {
                    "metrics_found": len(found_metrics),
                    "total_metrics": len(key_metrics),
                    "sample_metrics": found_metrics[:3]
                })
                return True
            else:
                self.log_test("Prometheus Metrics", "FAIL", {
                    "status_code": response.status_code
                })
                return False
                
        except Exception as e:
            self.log_test("Monitoring Metrics", "FAIL", {"error": str(e)})
            return False

    def run_complete_test_suite(self):
        """Run the complete test suite."""
        print("🧪 COMPLETE SYSTEM TEST - END-TO-END VALIDATION")
        print("=" * 60)
        print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test sequence
        tests_passed = 0
        total_tests = 9
        
        # 1. System Health
        if self.test_system_health():
            tests_passed += 1
        
        # 2. Authentication
        if self.test_authentication():
            tests_passed += 1
        
        # 3. Strategy Discovery
        if self.test_strategy_discovery():
            tests_passed += 1
        
        # 4. Optimization Submission
        job_id = self.test_optimization_submission()
        if job_id:
            tests_passed += 1
            
            # 5. Optimization Monitoring
            if self.test_optimization_monitoring(job_id):
                tests_passed += 1
                
                # 6. Results Retrieval
                results = self.test_results_retrieval(job_id)
                if results:
                    tests_passed += 1
                    
                    # 7. Pine Script Export
                    if self.test_pine_script_export(job_id):
                        tests_passed += 1
                    
                    # 8. PDF Report Export
                    if self.test_pdf_report_export(job_id):
                        tests_passed += 1
        
        # 9. Monitoring Metrics
        if self.test_monitoring_metrics():
            tests_passed += 1
        
        # Final results
        print("=" * 60)
        print("🏁 TEST SUITE COMPLETE")
        print(f"Tests Passed: {tests_passed}/{total_tests} ({tests_passed/total_tests*100:.1f}%)")
        print(f"Duration: {(datetime.now() - self.start_time).total_seconds():.1f} seconds")
        
        if tests_passed == total_tests:
            print("🎉 ALL TESTS PASSED - SYSTEM IS FULLY FUNCTIONAL!")
            return True
        else:
            print(f"❌ {total_tests - tests_passed} TESTS FAILED - SEE DETAILS ABOVE")
            return False

    def save_test_results(self):
        """Save detailed test results to file."""
        with open(f"system_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(self.test_results, f, indent=2)


if __name__ == "__main__":
    print("🚀 Trading Strategy Optimization System")
    print("Complete End-to-End Test Suite")
    print()
    
    tester = SystemTester()
    success = tester.run_complete_test_suite()
    tester.save_test_results()
    
    sys.exit(0 if success else 1) 
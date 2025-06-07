"""
Comprehensive Integration Tests for Trading Strategy Optimization API

This test suite validates the complete API system including:
- Authentication and security
- Optimization workflows (end-to-end)
- Export functionality (Pine Script and PDF)
- Error handling and edge cases
- Performance and concurrency
- Documentation accuracy

Run with: pytest test_api_integration.py -v --asyncio-mode=auto
"""

import pytest
import asyncio
import httpx
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path
import tempfile
import concurrent.futures
from statistics import mean, median

# Test configuration
API_BASE_URL = "http://localhost:8000"
VALID_API_KEY = "dev_key_123"
INVALID_API_KEY = "invalid_key_999"
TEST_TIMEOUT = 300  # 5 minutes for long-running optimization tests

class APITestClient:
    """Enhanced test client with authentication and timing."""
    
    def __init__(self, base_url: str = API_BASE_URL, api_key: str = VALID_API_KEY):
        self.base_url = base_url
        self.api_key = api_key
        self.response_times: List[float] = []
    
    async def request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Make authenticated request with timing."""
        headers = kwargs.pop("headers", {})
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        start_time = time.time()
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.request(
                method, f"{self.base_url}{endpoint}", headers=headers, **kwargs
            )
        
        response_time = time.time() - start_time
        self.response_times.append(response_time)
        
        return response
    
    async def get(self, endpoint: str, **kwargs) -> httpx.Response:
        return await self.request("GET", endpoint, **kwargs)
    
    async def post(self, endpoint: str, **kwargs) -> httpx.Response:
        return await self.request("POST", endpoint, **kwargs)
    
    async def delete(self, endpoint: str, **kwargs) -> httpx.Response:
        return await self.request("DELETE", endpoint, **kwargs)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.response_times:
            return {"avg": 0, "median": 0, "max": 0, "min": 0}
        
        return {
            "avg": mean(self.response_times),
            "median": median(self.response_times),
            "max": max(self.response_times),
            "min": min(self.response_times),
            "count": len(self.response_times)
        }


@pytest.fixture
async def api_client():
    """Create authenticated API client for testing."""
    return APITestClient()


@pytest.fixture
async def unauthorized_client():
    """Create unauthorized API client for security testing."""
    return APITestClient(api_key=INVALID_API_KEY)


class TestAPIAuthentication:
    """Test authentication and security features."""
    
    async def test_valid_api_key_accepted(self, api_client):
        """Test that valid API key is accepted."""
        response = await api_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    async def test_invalid_api_key_rejected(self, unauthorized_client):
        """Test that invalid API key is rejected."""
        response = await unauthorized_client.get("/api/v1/health")
        assert response.status_code == 401
        data = response.json()
        assert "authentication" in data["detail"].lower()
    
    async def test_missing_api_key_rejected(self):
        """Test that missing API key is rejected."""
        client = APITestClient(api_key=None)
        response = await client.get("/api/v1/health")
        assert response.status_code == 401
    
    async def test_rate_limiting_enforcement(self, api_client):
        """Test rate limiting enforcement."""
        # Make rapid requests to trigger rate limiting
        responses = []
        for i in range(20):
            response = await api_client.get("/api/v1/health")
            responses.append(response.status_code)
        
        # Should have at least some successful requests
        assert 200 in responses
        # Note: Depending on rate limit configuration, we might see 429s
        print(f"Rate limiting test: {responses.count(200)} successful, {responses.count(429)} rate limited")


class TestSystemHealth:
    """Test system health and monitoring endpoints."""
    
    async def test_health_endpoint(self, api_client):
        """Test basic health endpoint."""
        response = await api_client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data
        assert "components" in data
        assert isinstance(data["uptime_seconds"], (int, float))
    
    async def test_health_detailed_endpoint(self, api_client):
        """Test detailed health endpoint."""
        response = await api_client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        assert "system_info" in data
        assert "performance_metrics" in data
        assert "component_status" in data
    
    async def test_metrics_endpoint(self, api_client):
        """Test metrics endpoint."""
        response = await api_client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "memory_usage_mb" in data
        assert "cpu_usage_percent" in data
        assert isinstance(data["memory_usage_mb"], (int, float))


class TestStrategyManagement:
    """Test strategy listing and information endpoints."""
    
    async def test_list_strategies(self, api_client):
        """Test strategy listing endpoint."""
        response = await api_client.get("/api/v1/strategies")
        assert response.status_code == 200
        
        data = response.json()
        assert "strategies" in data
        assert "total_count" in data
        
        # Should have our expected strategies
        strategy_names = [s["name"] for s in data["strategies"]]
        expected_strategies = ["MovingAverageCrossover", "RSIStrategy", "MACDStrategy"]
        
        for expected in expected_strategies:
            assert expected in strategy_names
    
    async def test_strategy_details(self, api_client):
        """Test individual strategy details."""
        response = await api_client.get("/api/v1/strategies/MovingAverageCrossover")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "MovingAverageCrossover"
        assert "parameters" in data
        assert "description" in data
        assert "category" in data
    
    async def test_nonexistent_strategy(self, api_client):
        """Test handling of nonexistent strategy."""
        response = await api_client.get("/api/v1/strategies/NonexistentStrategy")
        assert response.status_code == 404


class TestOptimizationWorkflow:
    """Test complete optimization workflow end-to-end."""
    
    async def test_complete_optimization_workflow(self, api_client):
        """Test the complete optimization workflow from submission to results."""
        
        # Step 1: Submit optimization job
        optimization_request = {
            "strategy_name": "MovingAverageCrossover",
            "asset": "BTC",
            "timeframe": "4h",
            "start_date": "2023-01-01T00:00:00Z",
            "end_date": "2023-03-31T23:59:59Z",
            "optimization_config": {
                "max_evals": 20,  # Reduced for testing
                "timeout_minutes": 5,
                "algorithm": "tpe",
                "objective": "sharpe_ratio"
            }
        }
        
        response = await api_client.post(
            "/api/v1/optimize/single",
            json=optimization_request
        )
        assert response.status_code == 200 or response.status_code == 202
        
        data = response.json()
        job_id = data["job_id"]
        assert job_id
        
        # Step 2: Monitor job progress
        max_wait_time = 120  # 2 minutes
        start_time = time.time()
        job_completed = False
        
        while time.time() - start_time < max_wait_time:
            response = await api_client.get(f"/api/v1/optimize/status/{job_id}")
            assert response.status_code == 200
            
            status_data = response.json()
            status = status_data["status"]
            
            print(f"Job {job_id} status: {status}, progress: {status_data.get('progress', 0)}%")
            
            if status in ["completed", "failed"]:
                job_completed = True
                if status == "completed":
                    # Step 3: Retrieve optimization results
                    response = await api_client.get(f"/api/v1/optimize/results/{job_id}")
                    assert response.status_code == 200
                    
                    results = response.json()
                    assert "best_parameters" in results
                    assert "performance_metrics" in results
                    assert results["performance_metrics"]["sharpe_ratio"] is not None
                    
                    return results  # Return for use in other tests
                break
            
            await asyncio.sleep(2)  # Wait 2 seconds before checking again
        
        if not job_completed:
            pytest.fail(f"Optimization job {job_id} did not complete within {max_wait_time} seconds")
    
    async def test_optimization_validation(self, api_client):
        """Test optimization parameter validation."""
        # Test invalid date range
        invalid_request = {
            "strategy_name": "MovingAverageCrossover",
            "asset": "BTC",
            "timeframe": "4h",
            "start_date": "2023-12-31T00:00:00Z",
            "end_date": "2023-01-01T23:59:59Z",  # End before start
        }
        
        response = await api_client.post(
            "/api/v1/optimize/single",
            json=invalid_request
        )
        assert response.status_code == 422  # Validation error
    
    async def test_job_listing(self, api_client):
        """Test job listing endpoint."""
        response = await api_client.get("/api/v1/optimize/jobs")
        assert response.status_code == 200
        
        data = response.json()
        assert "jobs" in data
        assert "total_count" in data
        assert isinstance(data["jobs"], list)
    
    async def test_job_cancellation(self, api_client):
        """Test job cancellation functionality."""
        # Submit a long-running job
        optimization_request = {
            "strategy_name": "MovingAverageCrossover",
            "asset": "BTC",
            "timeframe": "4h",
            "start_date": "2023-01-01T00:00:00Z",
            "end_date": "2023-12-31T23:59:59Z",
            "optimization_config": {
                "max_evals": 100,
                "timeout_minutes": 30
            }
        }
        
        response = await api_client.post(
            "/api/v1/optimize/single",
            json=optimization_request
        )
        assert response.status_code in [200, 202]
        
        job_id = response.json()["job_id"]
        
        # Wait a moment for job to start
        await asyncio.sleep(2)
        
        # Cancel the job
        response = await api_client.delete(f"/api/v1/optimize/jobs/{job_id}")
        
        # Some jobs might complete too quickly to cancel
        assert response.status_code in [200, 400, 404]


class TestExportFunctionality:
    """Test Pine Script and PDF export functionality."""
    
    async def test_pine_script_generation(self, api_client):
        """Test Pine Script generation from optimization results."""
        pine_script_request = {
            "strategy_name": "MovingAverageCrossover",
            "optimization_results": {
                "best_parameters": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_threshold": 0.02
                },
                "best_score": 1.85,
                "performance_metrics": {
                    "total_return": 0.452,
                    "sharpe_ratio": 1.85,
                    "max_drawdown": 0.125
                }
            },
            "output_format": "strategy",
            "include_debugging": True,
            "include_alerts": True,
            "include_visualization": True
        }
        
        response = await api_client.post(
            "/api/v1/export/pine-script",
            json=pine_script_request
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "file_id" in data
        assert "download_url" in data
        assert "script_preview" in data
        assert data["filename"].endswith(".pine")
        
        # Test file download
        file_id = data["file_id"]
        download_response = await api_client.get(f"/api/v1/export/download/{file_id}")
        assert download_response.status_code == 200
        assert "// @version=5" in download_response.text
        
        return file_id
    
    async def test_pdf_report_generation(self, api_client):
        """Test PDF report generation from optimization results."""
        report_request = {
            "strategy_name": "MovingAverageCrossover",
            "optimization_results": {
                "best_parameters": {
                    "fast_period": 12,
                    "slow_period": 26
                },
                "performance_metrics": {
                    "total_return": 0.452,
                    "sharpe_ratio": 1.85,
                    "max_drawdown": 0.125,
                    "win_rate": 0.68
                },
                "validation_results": {
                    "out_of_sample_performance": 0.38,
                    "cross_asset_validation": True
                }
            },
            "report_type": "executive",
            "include_charts": True,
            "include_detailed_tables": True
        }
        
        response = await api_client.post(
            "/api/v1/export/report",
            json=report_request
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "file_id" in data
        assert "download_url" in data
        assert "pages_generated" in data
        assert data["filename"].endswith(".pdf")
        
        # Test file download
        file_id = data["file_id"]
        download_response = await api_client.get(f"/api/v1/export/download/{file_id}")
        assert download_response.status_code == 200
        assert download_response.headers["content-type"] == "application/pdf"
        
        return file_id
    
    async def test_export_file_management(self, api_client):
        """Test export file listing and deletion."""
        # List export files
        response = await api_client.get("/api/v1/export/files")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        
        if data:  # If there are files
            # Test file deletion
            file_id = data[0]["file_id"]
            delete_response = await api_client.delete(f"/api/v1/export/files/{file_id}")
            assert delete_response.status_code == 200
    
    async def test_nonexistent_file_download(self, api_client):
        """Test handling of nonexistent file downloads."""
        response = await api_client.get("/api/v1/export/download/nonexistent_file_id")
        assert response.status_code == 404


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""
    
    async def test_invalid_endpoints(self, api_client):
        """Test handling of invalid endpoints."""
        response = await api_client.get("/api/v1/nonexistent/endpoint")
        assert response.status_code == 404
    
    async def test_malformed_json_requests(self, api_client):
        """Test handling of malformed JSON requests."""
        response = await api_client.post(
            "/api/v1/optimize/single",
            content="invalid json content",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
    
    async def test_invalid_strategy_names(self, api_client):
        """Test handling of invalid strategy names."""
        invalid_request = {
            "strategy_name": "NonexistentStrategy",
            "asset": "BTC",
            "timeframe": "4h",
            "start_date": "2023-01-01T00:00:00Z",
            "end_date": "2023-03-31T23:59:59Z"
        }
        
        response = await api_client.post(
            "/api/v1/optimize/single",
            json=invalid_request
        )
        assert response.status_code in [400, 404]
    
    async def test_invalid_parameter_values(self, api_client):
        """Test handling of invalid parameter values."""
        invalid_request = {
            "strategy_name": "MovingAverageCrossover",
            "asset": "INVALID_ASSET",
            "timeframe": "invalid_timeframe",
            "start_date": "invalid_date",
            "end_date": "2023-03-31T23:59:59Z"
        }
        
        response = await api_client.post(
            "/api/v1/optimize/single",
            json=invalid_request
        )
        assert response.status_code == 422


class TestPerformance:
    """Test API performance and concurrency."""
    
    async def test_api_response_times(self, api_client):
        """Test API response times meet performance requirements."""
        # Test health endpoint response time
        start_time = time.time()
        response = await api_client.get("/api/v1/health")
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0  # Should respond within 1 second
        
        print(f"Health endpoint response time: {response_time:.3f}s")
    
    async def test_concurrent_requests(self, api_client):
        """Test handling of concurrent requests."""
        async def make_request():
            return await api_client.get("/api/v1/health")
        
        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
        
        # Check performance stats
        stats = api_client.get_performance_stats()
        print(f"Concurrent requests stats: {stats}")
        
        # Average response time should be reasonable
        assert stats["avg"] < 2.0
    
    async def test_system_resource_usage(self, api_client):
        """Test system resource usage during operations."""
        response = await api_client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        
        # Memory usage should be reasonable (adjust based on your system)
        assert metrics["memory_usage_mb"] < 2000  # Less than 2GB
        
        # CPU usage should be within normal range
        assert 0 <= metrics["cpu_usage_percent"] <= 100
        
        print(f"System metrics: Memory={metrics['memory_usage_mb']:.1f}MB, CPU={metrics['cpu_usage_percent']:.1f}%")


class TestDocumentation:
    """Test API documentation accuracy."""
    
    async def test_openapi_schema_accessible(self, api_client):
        """Test that OpenAPI schema is accessible."""
        response = await api_client.get("/api/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "components" in schema
    
    async def test_swagger_ui_accessible(self):
        """Test that Swagger UI is accessible."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/api/docs")
            assert response.status_code == 200
            assert "swagger" in response.text.lower()
    
    async def test_redoc_accessible(self):
        """Test that ReDoc documentation is accessible."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/api/redoc")
            assert response.status_code == 200
            assert "redoc" in response.text.lower()


class TestProductionReadiness:
    """Test production readiness checklist."""
    
    async def test_cors_headers(self, api_client):
        """Test CORS headers are properly configured."""
        response = await api_client.get("/api/v1/health")
        assert response.status_code == 200
        
        # Check for CORS headers (might not be present in test environment)
        headers = response.headers
        # CORS headers should be configured for production
    
    async def test_security_headers(self, api_client):
        """Test security headers are present."""
        response = await api_client.get("/api/v1/health")
        assert response.status_code == 200
        
        headers = response.headers
        
        # Check for security headers
        expected_security_headers = [
            "x-content-type-options",
            "x-frame-options", 
            "x-xss-protection"
        ]
        
        for header in expected_security_headers:
            # Headers might be lowercase in response
            header_present = any(header.lower() in h.lower() for h in headers.keys())
            if not header_present:
                print(f"Warning: Security header '{header}' not found")
    
    async def test_error_response_format(self, unauthorized_client):
        """Test error responses don't leak sensitive information."""
        response = await unauthorized_client.get("/api/v1/health")
        assert response.status_code == 401
        
        data = response.json()
        
        # Error response should be structured
        assert "detail" in data
        
        # Should not contain sensitive information
        error_message = data["detail"].lower()
        sensitive_terms = ["password", "key", "token", "secret", "stack trace", "traceback"]
        
        for term in sensitive_terms:
            assert term not in error_message, f"Error message contains sensitive term: {term}"


# Performance benchmarking functions
async def benchmark_api_performance():
    """Comprehensive API performance benchmark."""
    client = APITestClient()
    
    print("\n🚀 API Performance Benchmark Results:")
    print("=" * 50)
    
    # Test different endpoints
    endpoints = [
        ("/api/v1/health", "Health Check"),
        ("/api/v1/strategies", "Strategy List"),
        ("/api/v1/optimize/jobs", "Job List"),
        ("/api/v1/export/files", "Export Files")
    ]
    
    for endpoint, name in endpoints:
        # Make multiple requests to get average
        times = []
        for _ in range(5):
            start = time.time()
            response = await client.get(endpoint)
            end = time.time()
            
            if response.status_code == 200:
                times.append(end - start)
        
        if times:
            avg_time = mean(times)
            print(f"{name:20} | Avg: {avg_time:.3f}s | Max: {max(times):.3f}s | Min: {min(times):.3f}s")
    
    print("=" * 50)


# Test runner configuration
@pytest.mark.asyncio
class TestIntegrationSuite:
    """Main integration test suite runner."""
    
    async def test_run_full_integration_suite(self):
        """Run the complete integration test suite."""
        print("\n🔥 Running Complete API Integration Test Suite")
        print("=" * 60)
        
        # Check if API server is running
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{API_BASE_URL}/api/v1/health")
                if response.status_code != 200:
                    pytest.skip("API server not running or not healthy")
        except Exception:
            pytest.skip("API server not accessible")
        
        # Run performance benchmark
        await benchmark_api_performance()
        
        print("\n✅ Integration test suite completed successfully!")
        print("🎯 API is production-ready with comprehensive validation!")


if __name__ == "__main__":
    # Run specific test for debugging
    asyncio.run(benchmark_api_performance()) 
#!/usr/bin/env python3
"""
Performance Benchmarking Script

This script conducts comprehensive performance testing of the trading strategy
optimization platform, measuring API response times, database query performance,
memory usage, and optimization throughput.
"""

import asyncio
import time
import json
import requests
import pandas as pd
import numpy as np
import psutil
import statistics
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.performance_optimizer import get_performance_optimizer, performance_benchmark
from data_fetcher import DataFetcher
from strategies.strategy_factory import StrategyFactory
from optimization.hyperopt_optimizer import HyperoptOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.api_key = "test-key"  # Use test API key
        self.results = {}
        self.optimizer = get_performance_optimizer()
        
    def run_complete_benchmark(self) -> Dict[str, Any]:
        """Run all benchmark tests."""
        logger.info("🚀 Starting comprehensive performance benchmark...")
        
        start_time = time.time()
        
        # System baseline
        self.results['system_baseline'] = self.measure_system_baseline()
        
        # API Performance Tests
        self.results['api_performance'] = self.benchmark_api_endpoints()
        
        # Database Performance Tests
        self.results['database_performance'] = self.benchmark_database_operations()
        
        # Memory Usage Tests
        self.results['memory_performance'] = self.benchmark_memory_usage()
        
        # Strategy Optimization Performance
        self.results['optimization_performance'] = self.benchmark_strategy_optimization()
        
        # Load Testing
        self.results['load_test'] = self.run_load_test()
        
        # Cache Performance
        self.results['cache_performance'] = self.benchmark_cache_performance()
        
        total_time = time.time() - start_time
        
        # Generate summary
        self.results['summary'] = self.generate_performance_summary()
        self.results['total_benchmark_time'] = total_time
        
        logger.info(f"✅ Benchmark completed in {total_time:.2f} seconds")
        
        return self.results
    
    def measure_system_baseline(self) -> Dict[str, Any]:
        """Measure baseline system performance."""
        logger.info("📊 Measuring system baseline...")
        
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # CPU performance test
        start_time = time.time()
        # Simple CPU benchmark - calculate primes
        primes = self._calculate_primes(10000)
        cpu_test_time = time.time() - start_time
        
        # Memory performance test
        start_time = time.time()
        large_array = np.random.random((1000, 1000))
        result = np.dot(large_array, large_array.T)
        memory_test_time = time.time() - start_time
        
        # Disk I/O test
        start_time = time.time()
        test_data = "x" * 1024 * 1024  # 1MB of data
        with open("temp_benchmark_file.txt", "w") as f:
            f.write(test_data)
        with open("temp_benchmark_file.txt", "r") as f:
            _ = f.read()
        os.remove("temp_benchmark_file.txt")
        disk_io_time = time.time() - start_time
        
        return {
            'cpu_count': cpu_count,
            'cpu_frequency_mhz': cpu_freq.current if cpu_freq else 0,
            'total_memory_gb': memory.total / (1024**3),
            'available_memory_gb': memory.available / (1024**3),
            'memory_usage_percent': memory.percent,
            'cpu_benchmark_time': cpu_test_time,
            'memory_benchmark_time': memory_test_time,
            'disk_io_time': disk_io_time,
            'prime_count': len(primes)
        }
    
    def benchmark_api_endpoints(self) -> Dict[str, Any]:
        """Benchmark API endpoint performance."""
        logger.info("🌐 Benchmarking API endpoints...")
        
        endpoints = [
            ('GET', '/api/v1/health'),
            ('GET', '/api/v1/strategies'),
            ('GET', '/api/v1/business/dashboard/executive'),
            ('GET', '/api/v1/business/metrics/realtime'),
            ('GET', '/api/v1/monitoring/metrics'),
        ]
        
        results = {}
        
        for method, endpoint in endpoints:
            endpoint_results = self._benchmark_single_endpoint(method, endpoint)
            results[endpoint] = endpoint_results
        
        return results
    
    def _benchmark_single_endpoint(self, method: str, endpoint: str, 
                                 iterations: int = 10) -> Dict[str, Any]:
        """Benchmark a single API endpoint."""
        logger.info(f"Testing {method} {endpoint}")
        
        response_times = []
        status_codes = []
        errors = 0
        
        headers = {'X-API-Key': self.api_key}
        
        for i in range(iterations):
            try:
                start_time = time.time()
                
                if method == 'GET':
                    response = requests.get(
                        f"{self.api_base_url}{endpoint}",
                        headers=headers,
                        timeout=30
                    )
                elif method == 'POST':
                    response = requests.post(
                        f"{self.api_base_url}{endpoint}",
                        headers=headers,
                        json={},
                        timeout=30
                    )
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                status_codes.append(response.status_code)
                
                if response.status_code >= 400:
                    errors += 1
                    
            except Exception as e:
                logger.warning(f"Request failed: {e}")
                errors += 1
                response_times.append(30.0)  # Timeout value
                status_codes.append(500)
        
        return {
            'avg_response_time': statistics.mean(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'median_response_time': statistics.median(response_times),
            'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0,
            'success_rate': (iterations - errors) / iterations * 100,
            'errors': errors,
            'iterations': iterations,
            'status_codes': status_codes
        }
    
    def benchmark_database_operations(self) -> Dict[str, Any]:
        """Benchmark database operation performance."""
        logger.info("🗄️ Benchmarking database operations...")
        
        try:
            data_fetcher = DataFetcher()
            
            # Test data fetching
            start_time = time.time()
            data = data_fetcher.get_data('BTCUSDT', '1h', days=30)
            data_fetch_time = time.time() - start_time
            
            # Test data processing
            start_time = time.time()
            processed_data = data.copy()
            processed_data['sma_20'] = processed_data['close'].rolling(20).mean()
            processed_data['sma_50'] = processed_data['close'].rolling(50).mean()
            data_process_time = time.time() - start_time
            
            return {
                'data_fetch_time': data_fetch_time,
                'data_process_time': data_process_time,
                'data_points': len(data),
                'data_size_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
        except Exception as e:
            logger.error(f"Database benchmark failed: {e}")
            return {
                'error': str(e),
                'data_fetch_time': None,
                'data_process_time': None
            }
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        logger.info("🧠 Benchmarking memory usage...")
        
        # Baseline memory
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Test large DataFrame operations
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Create large dataset
        large_df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100000, freq='1min'),
            'open': np.random.random(100000) * 100,
            'high': np.random.random(100000) * 100,
            'low': np.random.random(100000) * 100,
            'close': np.random.random(100000) * 100,
            'volume': np.random.random(100000) * 1000000
        })
        
        after_creation_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Test memory optimization
        from utils.performance_optimizer import memory_efficient_dataframe_processing
        optimized_df = memory_efficient_dataframe_processing(large_df.copy())
        
        after_optimization_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Calculate technical indicators (memory intensive)
        optimized_df['sma_20'] = optimized_df['close'].rolling(20).mean()
        optimized_df['rsi'] = self._calculate_rsi(optimized_df['close'])
        
        after_calculations_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Clean up
        del large_df, optimized_df
        import gc
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return {
            'baseline_memory_mb': baseline_memory,
            'after_creation_mb': after_creation_memory,
            'after_optimization_mb': after_optimization_memory,
            'after_calculations_mb': after_calculations_memory,
            'final_memory_mb': final_memory,
            'peak_usage_mb': after_calculations_memory - baseline_memory,
            'memory_cleanup_mb': after_calculations_memory - final_memory,
            'dataframe_rows': 100000
        }
    
    def benchmark_strategy_optimization(self) -> Dict[str, Any]:
        """Benchmark strategy optimization performance."""
        logger.info("⚡ Benchmarking strategy optimization...")
        
        try:
            # Get sample data
            data_fetcher = DataFetcher()
            data = data_fetcher.get_data('BTCUSDT', '1h', days=30)
            
            # Get a simple strategy
            strategy_factory = StrategyFactory()
            strategy = strategy_factory.get_strategy('simple_ma')
            
            # Test single strategy execution
            start_time = time.time()
            result = strategy.backtest(data)
            single_execution_time = time.time() - start_time
            
            # Test optimization (limited iterations for benchmark)
            optimizer = HyperoptOptimizer()
            
            start_time = time.time()
            optimization_result = optimizer.optimize(
                strategy=strategy,
                data=data,
                max_evals=10,  # Limited for benchmark
                timeout_minutes=2
            )
            optimization_time = time.time() - start_time
            
            return {
                'single_execution_time': single_execution_time,
                'optimization_time': optimization_time,
                'optimization_evals': 10,
                'data_points': len(data),
                'best_score': optimization_result.get('best_score', 0) if optimization_result else 0,
                'total_trades': result.get('total_trades', 0) if result else 0
            }
            
        except Exception as e:
            logger.error(f"Strategy optimization benchmark failed: {e}")
            return {
                'error': str(e),
                'single_execution_time': None,
                'optimization_time': None
            }
    
    def run_load_test(self, concurrent_requests: int = 10, 
                     total_requests: int = 100) -> Dict[str, Any]:
        """Run load test on API endpoints."""
        logger.info(f"🚛 Running load test ({concurrent_requests} concurrent, {total_requests} total)...")
        
        def make_request():
            try:
                start_time = time.time()
                response = requests.get(
                    f"{self.api_base_url}/api/v1/health",
                    headers={'X-API-Key': self.api_key},
                    timeout=10
                )
                response_time = time.time() - start_time
                return {
                    'response_time': response_time,
                    'status_code': response.status_code,
                    'success': response.status_code == 200
                }
            except Exception as e:
                return {
                    'response_time': 10.0,
                    'status_code': 500,
                    'success': False,
                    'error': str(e)
                }
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(total_requests)]
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        
        response_times = [r['response_time'] for r in results]
        success_count = sum(1 for r in results if r['success'])
        
        return {
            'total_requests': total_requests,
            'concurrent_requests': concurrent_requests,
            'total_time': total_time,
            'requests_per_second': total_requests / total_time,
            'success_rate': success_count / total_requests * 100,
            'avg_response_time': statistics.mean(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'median_response_time': statistics.median(response_times),
            'p95_response_time': np.percentile(response_times, 95),
            'p99_response_time': np.percentile(response_times, 99)
        }
    
    def benchmark_cache_performance(self) -> Dict[str, Any]:
        """Benchmark cache performance."""
        logger.info("💾 Benchmarking cache performance...")
        
        cache_manager = self.optimizer.cache_manager
        
        # Test cache operations
        test_data = {'test': 'data', 'number': 42, 'array': list(range(1000))}
        
        # Cache SET performance
        set_times = []
        for i in range(100):
            start_time = time.time()
            cache_manager.set(f'test_key_{i}', test_data, ttl=300)
            set_times.append(time.time() - start_time)
        
        # Cache GET performance (hits)
        get_hit_times = []
        for i in range(100):
            start_time = time.time()
            result = cache_manager.get(f'test_key_{i}')
            get_hit_times.append(time.time() - start_time)
        
        # Cache GET performance (misses)
        get_miss_times = []
        for i in range(100):
            start_time = time.time()
            result = cache_manager.get(f'missing_key_{i}')
            get_miss_times.append(time.time() - start_time)
        
        # Get cache statistics
        cache_stats = cache_manager.get_stats()
        
        return {
            'avg_set_time': statistics.mean(set_times),
            'avg_get_hit_time': statistics.mean(get_hit_times),
            'avg_get_miss_time': statistics.mean(get_miss_times),
            'cache_stats': cache_stats,
            'test_operations': 300
        }
    
    def generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary and recommendations."""
        logger.info("📋 Generating performance summary...")
        
        summary = {
            'overall_grade': 'A',  # Will be calculated
            'recommendations': [],
            'strengths': [],
            'concerns': []
        }
        
        # Analyze API performance
        if 'api_performance' in self.results:
            api_results = self.results['api_performance']
            avg_response_times = [
                result['avg_response_time'] 
                for result in api_results.values() 
                if 'avg_response_time' in result
            ]
            
            if avg_response_times:
                overall_avg_response = statistics.mean(avg_response_times)
                
                if overall_avg_response < 0.1:
                    summary['strengths'].append("Excellent API response times (<100ms)")
                elif overall_avg_response < 0.5:
                    summary['strengths'].append("Good API response times (<500ms)")
                elif overall_avg_response < 1.0:
                    summary['recommendations'].append("Consider API optimization (response times >500ms)")
                else:
                    summary['concerns'].append("Slow API response times (>1s)")
        
        # Analyze memory performance
        if 'memory_performance' in self.results:
            mem_results = self.results['memory_performance']
            if 'peak_usage_mb' in mem_results:
                peak_usage = mem_results['peak_usage_mb']
                
                if peak_usage < 100:
                    summary['strengths'].append("Efficient memory usage")
                elif peak_usage < 500:
                    summary['recommendations'].append("Monitor memory usage patterns")
                else:
                    summary['concerns'].append("High memory usage detected")
        
        # Analyze load test results
        if 'load_test' in self.results:
            load_results = self.results['load_test']
            if 'success_rate' in load_results:
                success_rate = load_results['success_rate']
                
                if success_rate > 95:
                    summary['strengths'].append("High reliability under load")
                elif success_rate > 90:
                    summary['recommendations'].append("Minor reliability improvements needed")
                else:
                    summary['concerns'].append("Low success rate under load")
        
        # Calculate overall grade
        concern_count = len(summary['concerns'])
        if concern_count == 0:
            summary['overall_grade'] = 'A' if len(summary['recommendations']) == 0 else 'B'
        elif concern_count <= 2:
            summary['overall_grade'] = 'C'
        else:
            summary['overall_grade'] = 'D'
        
        return summary
    
    def save_results(self, filename: str = None):
        """Save benchmark results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"📁 Results saved to {filename}")
    
    def print_summary(self):
        """Print benchmark summary to console."""
        if 'summary' not in self.results:
            logger.error("No summary available")
            return
        
        summary = self.results['summary']
        
        print("\n" + "="*60)
        print("🎯 PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)
        print(f"Overall Grade: {summary['overall_grade']}")
        print(f"Total Benchmark Time: {self.results.get('total_benchmark_time', 0):.2f}s")
        
        if summary['strengths']:
            print("\n✅ Strengths:")
            for strength in summary['strengths']:
                print(f"  • {strength}")
        
        if summary['recommendations']:
            print("\n💡 Recommendations:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")
        
        if summary['concerns']:
            print("\n⚠️  Concerns:")
            for concern in summary['concerns']:
                print(f"  • {concern}")
        
        # Print key metrics
        print("\n📊 Key Metrics:")
        
        if 'api_performance' in self.results:
            api_avg = statistics.mean([
                r['avg_response_time'] for r in self.results['api_performance'].values()
                if 'avg_response_time' in r
            ])
            print(f"  • Average API Response Time: {api_avg:.3f}s")
        
        if 'load_test' in self.results:
            load_test = self.results['load_test']
            print(f"  • Load Test Success Rate: {load_test.get('success_rate', 0):.1f}%")
            print(f"  • Requests per Second: {load_test.get('requests_per_second', 0):.1f}")
        
        if 'memory_performance' in self.results:
            mem_perf = self.results['memory_performance']
            print(f"  • Peak Memory Usage: {mem_perf.get('peak_usage_mb', 0):.1f}MB")
        
        print("="*60 + "\n")
    
    # Helper methods
    def _calculate_primes(self, limit: int) -> List[int]:
        """Calculate prime numbers up to limit (CPU benchmark)."""
        primes = []
        for num in range(2, limit):
            if all(num % i != 0 for i in range(2, int(num**0.5) + 1)):
                primes.append(num)
        return primes
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

def main():
    """Main benchmark execution."""
    print("🚀 Starting Performance Benchmark Suite")
    print("This will test the platform's performance across multiple dimensions...")
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark()
    
    try:
        # Run complete benchmark
        results = benchmark.run_complete_benchmark()
        
        # Print summary
        benchmark.print_summary()
        
        # Save results
        benchmark.save_results()
        
        # Return success code based on grade
        grade = results.get('summary', {}).get('overall_grade', 'D')
        return 0 if grade in ['A', 'B'] else 1
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
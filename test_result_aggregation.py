#!/usr/bin/env python3
"""
Test Result Aggregation and Reporting System

This script tests the comprehensive result aggregation, analysis,
and reporting features for batch optimization results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from src.optimization.result_aggregator import ResultAggregator, BatchAnalysisReport
from src.optimization.result_visualizer import ResultVisualizer
from src.optimization.batch_manager import BatchOptimizationResult, BatchJobStatus
from src.optimization.hyperopt_optimizer import OptimizationResult

def create_mock_batch_job(name: str, score: float, sharpe: float, ret: float, i: int):
    """Create a mock batch job for testing."""
    mock_result = OptimizationResult(
        best_params={'param1': 0.5 + i * 0.1, 'param2': 10 + i * 2},
        best_score=score,
        best_metrics={
            'sharpe_ratio': sharpe,
            'annual_return': ret,
            'max_drawdown': 0.15 - i * 0.02,
            'total_trades': 150 - i * 10,
            'win_rate': 0.68 + i * 0.02
        },
        all_trials=[],
        optimization_time=25.5 + i * 3,
        total_evaluations=100,
        cache_hits=0
    )
    
    return BatchJobStatus(
        strategy_name=name,
        job_id=f"batch_test_job_{i}_{name}",
        status="completed",
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        result=mock_result
    )

def create_mock_batch_result() -> BatchOptimizationResult:
    """Create a comprehensive mock batch result for testing."""
    
    # Create mock optimization results
    strategies = [
        ("MovingAverageCrossover", 1.85, 1.85, 0.45),
        ("RSI", 1.72, 1.72, 0.38),
        ("MACD", 1.65, 1.65, 0.35),
        ("BollingerBands", 1.58, 1.58, 0.32),
        ("StochasticOscillator", 1.45, 1.45, 0.28)
    ]
    
    successful_jobs = []
    
    for i, (name, score, sharpe, ret) in enumerate(strategies):
        job = create_mock_batch_job(name, score, sharpe, ret, i)
        successful_jobs.append(job)
    
    # Create failed job for testing
    failed_job = BatchJobStatus(
        strategy_name="FailedStrategy",
        job_id="batch_test_job_failed",
        status="failed",
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        error_message="Mock failure for testing"
    )
    
    batch_result = BatchOptimizationResult(
        batch_id="batch_test_20241207_210000",
        total_strategies=6,
        successful_jobs=successful_jobs,
        failed_jobs=[failed_job],
        total_runtime_seconds=180.0,
        success_rate=83.33,  # 5/6 success
        summary_metrics={},
        resource_usage={
            'cpu_percent': 28.5,
            'memory_gb': 5.2,
            'monitoring_enabled': True
        }
    )
    
    return batch_result

def test_result_aggregator():
    """Test the ResultAggregator functionality."""
    print("🧪 Testing ResultAggregator...")
    
    aggregator = ResultAggregator()
    
    # Create test batch jobs
    strategies = [
        ("MovingAverageCrossover", 1.85, 1.85, 0.45),
        ("RSI", 1.72, 1.72, 0.38),
        ("MACD", 1.65, 1.65, 0.35),
        ("BollingerBands", 1.58, 1.58, 0.32),
        ("StochasticOscillator", 1.45, 1.45, 0.28)
    ]
    
    successful_jobs = []
    for i, (name, score, sharpe, ret) in enumerate(strategies):
        job = create_mock_batch_job(name, score, sharpe, ret, i)
        successful_jobs.append(job)
    
    # Test batch aggregation
    try:
        analysis = aggregator.aggregate_batch_results(
            successful_jobs=successful_jobs,
            failed_jobs=[],
            batch_id="test_batch_20241207",
            total_runtime=150.0
        )
        
        print("✅ Batch analysis generated successfully")
        print(f"   📊 Strategy Rankings: {len(analysis.strategy_rankings)} strategies")
        print(f"   📈 Top Performers: {len(analysis.top_performers)} strategies")
        print(f"   ⚠️  Poor Performers: {len(analysis.poor_performers)} strategies")
        print(f"   🎯 Recommendations: {len(analysis.recommendations)} insights")
        print(f"   📋 Success Rate: {analysis.success_rate:.1f}%")
        
        # Test to_dict conversion
        analysis_dict = analysis.to_dict()
        print(f"   🔄 Dictionary conversion: {type(analysis_dict)} with {len(analysis_dict)} keys")
        
        return True
        
    except Exception as e:
        print(f"❌ ResultAggregator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_result_visualizer():
    """Test the ResultVisualizer functionality."""
    print("\n🖼️  Testing ResultVisualizer...")
    
    visualizer = ResultVisualizer()
    batch_result = create_mock_batch_result()
    
    try:
        # Test summary report
        summary_report = visualizer.generate_summary_report(batch_result)
        print("✅ Summary report generated successfully")
        print(f"   📝 Report length: {len(summary_report)} characters")
        
        # Test detailed report
        detailed_report = visualizer.generate_detailed_report(batch_result)
        print("✅ Detailed report generated successfully")
        print(f"   📋 Report length: {len(detailed_report)} characters")
        
        # Test CSV export
        csv_export = visualizer.generate_csv_export(batch_result)
        print("✅ CSV export generated successfully")
        csv_lines = csv_export.split('\n')
        print(f"   📊 CSV rows: {len(csv_lines)} (including header)")
        
        # Test JSON export
        json_export = visualizer.generate_json_export(batch_result)
        print("✅ JSON export generated successfully")
        print(f"   🔗 JSON length: {len(json_export)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ ResultVisualizer test failed: {e}")
        return False

def test_enhanced_batch_result():
    """Test the enhanced BatchOptimizationResult functionality."""
    print("\n📊 Testing Enhanced BatchOptimizationResult...")
    
    batch_result = create_mock_batch_result()
    
    try:
        # Test top performers
        top_performers = batch_result.get_top_performers(3)
        print(f"✅ Top performers: {len(top_performers)} strategies")
        for i, performer in enumerate(top_performers, 1):
            print(f"   {i}. {performer['strategy_name']}: {performer['score']:.4f}")
        
        # Test optimization insights
        insights = batch_result.get_optimization_insights()
        print(f"✅ Optimization insights: {len(insights)} metrics")
        print(f"   🎯 Success rate: {insights['success_rate']:.1f}%")
        print(f"   ⏱️  Total runtime: {insights['total_runtime_hours']:.2f} hours")
        print(f"   📈 Strategies tested: {insights['strategies_tested']}")
        
        # Test to_dict conversion
        result_dict = batch_result.to_dict()
        print(f"✅ Dictionary conversion: {len(result_dict)} keys")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced BatchOptimizationResult test failed: {e}")
        return False

def test_integration():
    """Test full integration of aggregation and reporting."""
    print("\n🔗 Testing Full Integration...")
    
    try:
        # Create batch result
        batch_result = create_mock_batch_result()
        
        # Generate comprehensive analysis using aggregator
        aggregator = ResultAggregator()
        
        analysis_report = aggregator.aggregate_batch_results(
            successful_jobs=batch_result.successful_jobs,
            failed_jobs=batch_result.failed_jobs,
            batch_id=batch_result.batch_id,
            total_runtime=batch_result.total_runtime_seconds
        )
        
        # Update batch result with analysis
        batch_result.analysis_report = analysis_report.to_dict()
        batch_result.summary_metrics = {
            'comprehensive_analysis': analysis_report.to_dict(),
            'basic_stats': {'note': 'Integration test'}
        }
        
        # Generate all report types
        visualizer = ResultVisualizer()
        
        summary_report = visualizer.generate_summary_report(batch_result)
        detailed_report = visualizer.generate_detailed_report(batch_result)
        csv_export = visualizer.generate_csv_export(batch_result)
        json_export = visualizer.generate_json_export(batch_result)
        
        print("✅ Full integration test successful")
        print(f"   📊 Analysis report: {len(analysis_report.performance_rankings)} rankings")
        print(f"   📝 Summary report: {len(summary_report)} chars")
        print(f"   📋 Detailed report: {len(detailed_report)} chars")
        print(f"   📊 CSV export: {csv_export.count(',') + 1} columns")
        print(f"   🔗 JSON export: {len(json_export)} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Result Aggregation and Reporting System")
    print("=" * 60)
    
    tests = [
        test_result_aggregator,
        test_result_visualizer,
        test_enhanced_batch_result,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Result aggregation and reporting system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    exit(main()) 
"""
Result Visualization and Reporting

This module provides tools for creating human-readable reports
and summaries from batch optimization results.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Use absolute imports
try:
    from src.optimization.batch_manager import BatchOptimizationResult, BatchJobStatus
    from src.optimization.result_aggregator import BatchAnalysisReport
    from src.utils.logger import get_logger
except ImportError:
    # Fallback for relative imports
    from ..optimization.batch_manager import BatchOptimizationResult, BatchJobStatus
    from ..optimization.result_aggregator import BatchAnalysisReport
    from ..utils.logger import get_logger


class ResultVisualizer:
    """Generate formatted reports and summaries from batch optimization results."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def generate_summary_report(self, batch_result: BatchOptimizationResult) -> str:
        """Generate a concise summary report for batch optimization results."""
        
        report_lines = [
            "=" * 80,
            f"BATCH OPTIMIZATION SUMMARY - {batch_result.batch_id}",
            "=" * 80,
            "",
            f"📊 Overall Results:",
            f"   • Total Strategies: {batch_result.total_strategies}",
            f"   • Successful: {len(batch_result.successful_jobs)} ({batch_result.success_rate:.1f}%)",
            f"   • Failed: {len(batch_result.failed_jobs)}",
            f"   • Runtime: {batch_result.total_runtime_seconds / 60:.1f} minutes",
            ""
        ]
        
        # Add top performers
        top_performers = batch_result.get_top_performers(3)
        if top_performers:
            report_lines.extend([
                "🏆 Top Performing Strategies:",
                ""
            ])
            
            for i, performer in enumerate(top_performers, 1):
                strategy_name = performer.get('strategy_name', 'Unknown')
                score = performer.get('score', 0)
                metrics = performer.get('metrics', {})
                
                report_lines.append(f"   {i}. {strategy_name}")
                report_lines.append(f"      Score: {score:.4f}")
                
                if 'sharpe_ratio' in metrics:
                    report_lines.append(f"      Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                if 'annual_return' in metrics:
                    report_lines.append(f"      Annual Return: {metrics['annual_return']:.1%}")
                if 'max_drawdown' in metrics:
                    report_lines.append(f"      Max Drawdown: {metrics['max_drawdown']:.1%}")
                
                report_lines.append("")
        
        # Add resource usage
        if batch_result.resource_usage:
            report_lines.extend([
                "💻 Resource Usage:",
                ""
            ])
            
            for key, value in batch_result.resource_usage.items():
                if isinstance(value, (int, float)):
                    if 'cpu' in key.lower():
                        report_lines.append(f"   • {key}: {value:.1f}%")
                    elif 'memory' in key.lower() or 'gb' in key.lower():
                        report_lines.append(f"   • {key}: {value:.1f} GB")
                    else:
                        report_lines.append(f"   • {key}: {value}")
                else:
                    report_lines.append(f"   • {key}: {value}")
            
            report_lines.append("")
        
        # Add insights from comprehensive analysis
        if batch_result.analysis_report:
            insights = batch_result.analysis_report.get('insights', {})
            if insights:
                report_lines.extend([
                    "💡 Key Insights:",
                    ""
                ])
                
                for insight_key, insight_value in insights.items():
                    if isinstance(insight_value, str):
                        report_lines.append(f"   • {insight_key}: {insight_value}")
                    elif isinstance(insight_value, (int, float)):
                        report_lines.append(f"   • {insight_key}: {insight_value:.3f}")
                
                report_lines.append("")
        
        # Add failed jobs summary if any
        if batch_result.failed_jobs:
            report_lines.extend([
                "❌ Failed Jobs:",
                ""
            ])
            
            for job in batch_result.failed_jobs:
                report_lines.append(f"   • {job.strategy_name}: {job.error_message or 'Unknown error'}")
            
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def generate_detailed_report(self, batch_result: BatchOptimizationResult) -> str:
        """Generate a comprehensive detailed report."""
        
        report_lines = [
            "=" * 100,
            f"DETAILED BATCH OPTIMIZATION REPORT - {batch_result.batch_id}",
            "=" * 100,
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "📋 Batch Overview:",
            f"   Batch ID: {batch_result.batch_id}",
            f"   Total Strategies: {batch_result.total_strategies}",
            f"   Success Rate: {batch_result.success_rate:.2f}%",
            f"   Total Runtime: {batch_result.total_runtime_seconds:.2f} seconds ({batch_result.total_runtime_seconds / 3600:.2f} hours)",
            ""
        ]
        
        # Detailed successful jobs
        if batch_result.successful_jobs:
            report_lines.extend([
                "✅ Successful Optimizations:",
                ""
            ])
            
            for job in batch_result.successful_jobs:
                report_lines.append(f"Strategy: {job.strategy_name}")
                report_lines.append(f"   Job ID: {job.job_id}")
                report_lines.append(f"   Status: {job.status}")
                
                if job.start_time and job.end_time:
                    runtime = (job.end_time - job.start_time).total_seconds()
                    report_lines.append(f"   Runtime: {runtime:.2f} seconds")
                
                if job.result:
                    report_lines.append(f"   Best Score: {job.result.best_score:.6f}")
                    report_lines.append(f"   Evaluations: {job.result.total_evaluations}")
                    report_lines.append(f"   Optimization Time: {job.result.optimization_time:.2f}s")
                    
                    if job.result.best_metrics:
                        report_lines.append("   Metrics:")
                        for metric, value in job.result.best_metrics.items():
                            if isinstance(value, (int, float)):
                                report_lines.append(f"      {metric}: {value:.4f}")
                            else:
                                report_lines.append(f"      {metric}: {value}")
                    
                    if job.result.best_params:
                        report_lines.append("   Best Parameters:")
                        for param, value in job.result.best_params.items():
                            if isinstance(value, (int, float)):
                                report_lines.append(f"      {param}: {value:.4f}")
                            else:
                                report_lines.append(f"      {param}: {value}")
                
                report_lines.append("")
        
        # Failed jobs details
        if batch_result.failed_jobs:
            report_lines.extend([
                "❌ Failed Optimizations:",
                ""
            ])
            
            for job in batch_result.failed_jobs:
                report_lines.append(f"Strategy: {job.strategy_name}")
                report_lines.append(f"   Job ID: {job.job_id}")
                report_lines.append(f"   Status: {job.status}")
                report_lines.append(f"   Error: {job.error_message or 'Unknown error'}")
                
                if job.start_time and job.end_time:
                    runtime = (job.end_time - job.start_time).total_seconds()
                    report_lines.append(f"   Runtime before failure: {runtime:.2f} seconds")
                
                report_lines.append("")
        
        # Resource usage details
        if batch_result.resource_usage:
            report_lines.extend([
                "💻 Resource Usage Details:",
                ""
            ])
            
            for key, value in batch_result.resource_usage.items():
                report_lines.append(f"   {key}: {value}")
            
            report_lines.append("")
        
        # Comprehensive analysis details
        if batch_result.analysis_report:
            report_lines.extend([
                "📊 Comprehensive Analysis:",
                ""
            ])
            
            # Add formatted JSON for detailed analysis
            try:
                formatted_analysis = json.dumps(batch_result.analysis_report, indent=2)
                report_lines.append(formatted_analysis)
            except Exception as e:
                report_lines.append(f"   Error formatting analysis: {e}")
                report_lines.append(f"   Raw data: {batch_result.analysis_report}")
            
            report_lines.append("")
        
        report_lines.append("=" * 100)
        
        return "\n".join(report_lines)
    
    def generate_csv_export(self, batch_result: BatchOptimizationResult) -> str:
        """Generate CSV format for successful optimization results."""
        
        # CSV Header
        csv_lines = [
            "strategy_name,job_id,status,best_score,optimization_time,total_evaluations," +
            "sharpe_ratio,annual_return,max_drawdown,total_trades,win_rate," +
            "start_time,end_time,runtime_seconds"
        ]
        
        # Add successful jobs
        for job in batch_result.successful_jobs:
            if job.result:
                metrics = job.result.best_metrics or {}
                
                row = [
                    job.strategy_name,
                    job.job_id,
                    job.status,
                    str(job.result.best_score),
                    str(job.result.optimization_time),
                    str(job.result.total_evaluations),
                    str(metrics.get('sharpe_ratio', '')),
                    str(metrics.get('annual_return', '')),
                    str(metrics.get('max_drawdown', '')),
                    str(metrics.get('total_trades', '')),
                    str(metrics.get('win_rate', '')),
                    job.start_time.isoformat() if job.start_time else '',
                    job.end_time.isoformat() if job.end_time else '',
                    str(job.runtime_seconds or '')
                ]
                
                csv_lines.append(",".join(row))
        
        return "\n".join(csv_lines)
    
    def generate_json_export(self, batch_result: BatchOptimizationResult) -> str:
        """Generate JSON export of the complete batch result."""
        
        try:
            return json.dumps(batch_result.to_dict(), indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to generate JSON export: {e}")
            return json.dumps({"error": f"Export failed: {e}"}, indent=2) 
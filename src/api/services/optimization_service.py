"""
Optimization Service

Core business logic for managing optimization requests and coordinating
with the hyperparameter optimization engine and job management system.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
from fastapi import HTTPException

from ..models import (
    OptimizationRequest, OptimizationResult, OptimizationStatus, 
    BatchOptimizationRequest, JobInfo, PerformanceMetrics, Asset, TimeFrame
)
from ..job_manager import job_manager, JobPriority

# Import custom exceptions
try:
    from src.optimization.exceptions import (
        OptimizationError, InvalidStrategyError, OptimizationTimeoutError,
        BatchOptimizationError, ParameterValidationError, DataValidationError,
        ResourceExhaustionError, SerializationError, ConcurrencyError
    )
except ImportError:
    # Fallback if exceptions module is not available
    class OptimizationError(Exception):
        def to_dict(self):
            return {"error": str(self)}
# from src.data.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)

# Simple strategy mapping - we'll expand this as needed
AVAILABLE_STRATEGIES = {
    "MovingAverageCrossover": "src.strategies.moving_average_crossover",
    "RSI": "src.strategies.rsi_strategy", 
    "MACDStrategy": "src.strategies.macd_strategy",
    "BollingerBandsStrategy": "src.strategies.bollinger_bands_strategy",
    "MomentumStrategy": "src.strategies.momentum_strategy",
    "SimpleMAStrategy": "src.strategies.simple_ma_strategy",
    "VWAPStrategy": "src.strategies.vwap_strategy",
    "OBVStrategy": "src.strategies.obv_strategy",
    "ADStrategy": "src.strategies.ad_strategy",
    "CMFStrategy": "src.strategies.cmf_strategy",
    "ATRStrategy": "src.strategies.atr_strategy",
    "BollingerSqueezeStrategy": "src.strategies.bollinger_squeeze_strategy",
    "KeltnerChannelStrategy": "src.strategies.keltner_channel_strategy",
    "HistoricalVolatilityStrategy": "src.strategies.historical_volatility_strategy",
    "ROCStrategy": "src.strategies.roc_strategy",
    "StochasticStrategy": "src.strategies.stochastic_strategy",
    "WilliamsRStrategy": "src.strategies.williams_r_strategy",
    "SupportResistanceStrategy": "src.strategies.support_resistance_strategy",
    "PivotPointsStrategy": "src.strategies.pivot_points_strategy",
    "FibonacciRetracementStrategy": "src.strategies.fibonacci_retracement_strategy",
    "DoubleTopBottomStrategy": "src.strategies.double_top_bottom_strategy",
    "MTFRSIStrategy": "src.strategies.mtf_rsi_strategy",
    "MTFMACDStrategy": "src.strategies.mtf_macd_strategy",
    "MTFTrendAnalysisStrategy": "src.strategies.mtf_trend_analysis_strategy",
    "UltimateOscillatorStrategy": "src.strategies.ultimate_oscillator_strategy"
}

class OptimizationService:
    """
    Service for handling optimization operations and integrating with
    the core optimization engine.
    """
    
    def __init__(self):
        # Initialize core components
        # Create minimal config for DataFetcher (commented out for now)
        # self.data_fetcher = DataFetcher(data_config)
        
        # Cache for validation results
        self._strategy_cache = {}
        
        logger.info("OptimizationService initialized")
    
    async def submit_single_optimization(self, request: OptimizationRequest) -> Dict[str, Any]:
        """
        Submit a single strategy optimization job.
        
        Args:
            request: Optimization request configuration
            
        Returns:
            Job submission response with job_id and initial status
            
        Raises:
            ValueError: If validation fails
        """
        try:
            # Validate request
            await self._validate_optimization_request(request)
            
            # Submit job to job manager
            job_id = await job_manager.submit_job(request, JobPriority.NORMAL)
            
            logger.info(f"Submitted optimization job {job_id} for strategy {request.strategy_name}")
            
            return {
                "job_id": job_id,
                "status": "queued",
                "message": f"Optimization job submitted for strategy {request.strategy_name}",
                "estimated_start": self._estimate_job_start_time(),
                "configuration": {
                    "strategy": request.strategy_name,
                    "asset": request.asset.value,
                    "timeframe": request.timeframe.value,
                    "max_evaluations": request.optimization_config.max_evals,
                    "timeout_minutes": request.optimization_config.timeout_minutes
                }
            }
            
        except InvalidStrategyError as strategy_error:
            logger.error(f"Invalid strategy error: {strategy_error.to_dict()}")
            raise HTTPException(
                status_code=400,
                detail=strategy_error.to_dict()
            )
            
        except ParameterValidationError as param_error:
            logger.error(f"Parameter validation error: {param_error.to_dict()}")
            raise HTTPException(
                status_code=422,
                detail=param_error.to_dict()
            )
            
        except DataValidationError as data_error:
            logger.error(f"Data validation error: {data_error.to_dict()}")
            raise HTTPException(
                status_code=400,
                detail=data_error.to_dict()
            )
            
        except OptimizationError as opt_error:
            logger.error(f"Optimization error: {opt_error.to_dict()}")
            raise HTTPException(
                status_code=400,
                detail=opt_error.to_dict()
            )
            
        except Exception as e:
            logger.error(f"Unexpected error submitting optimization job: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "error_type": "InternalServerError",
                    "message": "An unexpected error occurred while submitting optimization",
                    "error_code": "INTERNAL_ERROR"
                }
            )
    
    async def submit_batch_optimization(self, request: BatchOptimizationRequest) -> Dict[str, Any]:
        """
        Submit multiple strategy optimization jobs using robust parallel execution.
        
        Args:
            request: Batch optimization request
            
        Returns:
            Batch optimization results with comprehensive metrics
        """
        try:
            # Import the new BatchOptimizationManager
            from src.optimization.batch_manager import BatchOptimizationManager, BatchOptimizationConfig
            
            # Configure batch optimization with robust settings
            batch_config = BatchOptimizationConfig(
                max_parallel_jobs=request.parallel_jobs,
                global_timeout_minutes=240,  # 4 hours total
                job_timeout_minutes=60,      # 1 hour per job
                resource_monitoring=True,
                retry_failed_jobs=True,
                max_retries=2
            )
            
            # Create batch manager and execute
            batch_manager = BatchOptimizationManager(batch_config)
            
            logger.info(f"Starting robust parallel batch optimization for {len(request.strategies)} strategies")
            
            # Execute batch optimization with true parallel processing
            batch_result = await batch_manager.execute_batch_optimization(request)
            
            # Format response for API compatibility
            submitted_jobs = []
            failed_submissions = []
            
            # Process successful jobs
            for job in batch_result.successful_jobs:
                submitted_jobs.append({
                    "strategy_name": job.strategy_name,
                    "job_id": job.job_id,
                    "status": job.status,
                    "runtime_seconds": job.runtime_seconds,
                    "best_score": job.result.best_score if job.result else None,
                    "progress": job.progress
                })
            
            # Process failed jobs
            for job in batch_result.failed_jobs:
                failed_submissions.append({
                    "strategy_name": job.strategy_name,
                    "job_id": job.job_id,
                    "status": job.status,
                    "error": job.error_message,
                    "retry_count": job.retry_count
                })
            
            logger.info(
                f"Robust batch optimization completed: "
                f"{len(submitted_jobs)} successful, {len(failed_submissions)} failed, "
                f"success rate: {batch_result.success_rate:.1f}%"
            )
            
            return {
                "batch_id": batch_result.batch_id,
                "total_strategies": batch_result.total_strategies,
                "submitted_jobs": submitted_jobs,
                "failed_submissions": failed_submissions,
                "success_rate": batch_result.success_rate,
                "total_runtime_seconds": batch_result.total_runtime_seconds,
                "summary_metrics": batch_result.summary_metrics,
                "resource_usage": batch_result.resource_usage,
                "parallel_execution": {
                    "max_parallel_jobs": batch_config.max_parallel_jobs,
                    "resource_monitoring": batch_config.resource_monitoring,
                    "timeout_configuration": {
                        "global_timeout_minutes": batch_config.global_timeout_minutes,
                        "job_timeout_minutes": batch_config.job_timeout_minutes
                    }
                }
            }
            
        except BatchOptimizationError as batch_error:
            # Handle batch optimization specific errors
            logger.error(f"Batch optimization error: {batch_error.to_dict()}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error_type": "BatchOptimizationError", 
                    "message": batch_error.message,
                    "error_code": batch_error.error_code,
                    "context": batch_error.context
                }
            )
            
        except ResourceExhaustionError as resource_error:
            # Handle resource exhaustion
            logger.error(f"Resource exhaustion in batch optimization: {resource_error.to_dict()}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error_type": "ResourceExhaustionError",
                    "message": resource_error.message,
                    "error_code": resource_error.error_code,
                    "retry_after": 300  # Suggest retry after 5 minutes
                }
            )
            
        except ConcurrencyError as concurrency_error:
            # Handle concurrency issues
            logger.error(f"Concurrency error in batch optimization: {concurrency_error.to_dict()}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error_type": "ConcurrencyError",
                    "message": concurrency_error.message,
                    "error_code": concurrency_error.error_code,
                    "retry_after": 60  # Suggest retry after 1 minute
                }
            )
            
        except OptimizationError as opt_error:
            # Handle any other optimization errors
            logger.error(f"Optimization error in batch: {opt_error.to_dict()}")
            raise HTTPException(
                status_code=400,
                detail=opt_error.to_dict()
            )
            
        except Exception as e:
            logger.error(f"Unexpected error in batch optimization: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "error_type": "InternalServerError",
                    "message": "An unexpected error occurred during batch optimization",
                    "error_code": "INTERNAL_ERROR"
                }
            )
    
    def get_optimization_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of an optimization job.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Job status information or None if not found
        """
        status = job_manager.get_job_status(job_id)
        
        if not status:
            return None
        
        # Enhance status with additional information
        enhanced_status = status.copy()
        enhanced_status.update({
            "queue_position": self._get_queue_position(job_id),
            "system_load": job_manager.get_queue_stats()["system_load"],
            "optimization_phase": self._determine_optimization_phase(status["progress"])
        })
        
        return enhanced_status
    
    def get_optimization_result(self, job_id: str) -> Optional[OptimizationResult]:
        """
        Get the results of a completed optimization job.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Optimization results or None if not found/not completed
        """
        return job_manager.get_job_result(job_id)
    
    def cancel_optimization(self, job_id: str) -> bool:
        """
        Cancel a queued or running optimization job.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        return job_manager.cancel_job(job_id)
    
    def list_optimization_jobs(self, 
                             status: Optional[OptimizationStatus] = None,
                             limit: int = 50) -> List[Dict[str, Any]]:
        """
        List optimization jobs with optional filtering.
        
        Args:
            status: Optional status filter
            limit: Maximum number of jobs to return
            
        Returns:
            List of job information dictionaries
        """
        return job_manager.list_jobs(status, limit)
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """
        Get overall optimization system statistics.
        
        Returns:
            System statistics and performance metrics
        """
        queue_stats = job_manager.get_queue_stats()
        
        return {
            "queue_statistics": queue_stats,
            "performance_metrics": {
                "average_completion_time_minutes": queue_stats["average_completion_time_minutes"],
                "success_rate_percent": queue_stats["success_rate"],
                "throughput_jobs_per_hour": self._calculate_throughput(),
                "resource_utilization": queue_stats["system_load"]
            },
            "capacity": {
                "max_concurrent_jobs": job_manager.max_concurrent_jobs,
                "max_queue_size": job_manager.max_queue_size,
                "current_utilization_percent": (
                    queue_stats["active_jobs"] / job_manager.max_concurrent_jobs * 100
                )
            }
        }
    
    async def _validate_optimization_request(self, request: OptimizationRequest):
        """
        Validate optimization request parameters.
        
        Args:
            request: Optimization request to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check if strategy exists
        if request.strategy_name not in AVAILABLE_STRATEGIES:
            raise ValueError(f"Strategy '{request.strategy_name}' not found")
        
        # Validate date range
        if request.end_date <= request.start_date:
            raise ValueError("end_date must be after start_date")
        
        # Check if date range is reasonable
        date_range = (request.end_date - request.start_date).days
        if date_range < 30:
            raise ValueError("Date range must be at least 30 days for meaningful optimization")
        
        if date_range > 1825:  # 5 years
            raise ValueError("Date range cannot exceed 5 years")
        
        # Validate optimization configuration
        if request.optimization_config.max_evals < 10:
            raise ValueError("max_evals must be at least 10")
        
        if request.optimization_config.max_evals > 1000:
            raise ValueError("max_evals cannot exceed 1000")
        
        if request.optimization_config.timeout_minutes < 5:
            raise ValueError("timeout_minutes must be at least 5")
        
        if request.optimization_config.timeout_minutes > 480:  # 8 hours
            raise ValueError("timeout_minutes cannot exceed 480 (8 hours)")
        
        # Check data availability (basic check)
        try:
            # This would be a quick check if data exists for the requested period
            # In a full implementation, this would query the data fetcher
            logger.debug(f"Validating data availability for {request.asset.value} "
                        f"from {request.start_date} to {request.end_date}")
        except Exception as e:
            raise ValueError(f"Data validation failed: {str(e)}")
    
    def _estimate_job_start_time(self) -> str:
        """Estimate when a newly submitted job will start."""
        queue_stats = job_manager.get_queue_stats()
        
        # Estimate based on queue size and average completion time
        queue_size = queue_stats["queue_size"]
        active_jobs = queue_stats["active_jobs"]
        avg_completion_time = queue_stats["average_completion_time_minutes"]
        
        if active_jobs < job_manager.max_concurrent_jobs:
            # Job will start immediately
            estimated_start = datetime.utcnow()
        else:
            # Estimate based on queue position and completion times
            position_in_queue = queue_size
            estimated_wait_minutes = (position_in_queue / job_manager.max_concurrent_jobs) * avg_completion_time
            estimated_start = datetime.utcnow() + timedelta(minutes=estimated_wait_minutes)
        
        return estimated_start.isoformat()
    
    def _estimate_batch_completion(self, job_count: int) -> str:
        """Estimate completion time for a batch of jobs."""
        queue_stats = job_manager.get_queue_stats()
        avg_completion_time = queue_stats["average_completion_time_minutes"]
        
        # Estimate based on parallel processing capability
        parallel_capacity = job_manager.max_concurrent_jobs
        estimated_batches = (job_count + parallel_capacity - 1) // parallel_capacity
        total_estimated_time = estimated_batches * avg_completion_time
        
        estimated_completion = datetime.utcnow() + timedelta(minutes=total_estimated_time)
        return estimated_completion.isoformat()
    
    def _get_queue_position(self, job_id: str) -> Optional[int]:
        """Get the position of a job in the queue."""
        # This would require tracking queue positions in the job manager
        # For now, return None (not implemented)
        return None
    
    def _determine_optimization_phase(self, progress: float) -> str:
        """Determine current optimization phase based on progress."""
        if progress < 10:
            return "initializing"
        elif progress < 20:
            return "data_preparation"
        elif progress < 70:
            return "hyperparameter_optimization"
        elif progress < 85:
            return "backtesting"
        elif progress < 95:
            return "performance_analysis"
        elif progress < 100:
            return "finalizing"
        else:
            return "completed"
    
    def _calculate_throughput(self) -> float:
        """Calculate system throughput in jobs per hour."""
        queue_stats = job_manager.get_queue_stats()
        
        if queue_stats["average_completion_time_minutes"] > 0:
            jobs_per_hour = (60 / queue_stats["average_completion_time_minutes"]) * job_manager.max_concurrent_jobs
            return round(jobs_per_hour, 2)
        
        return 0.0
    
    def get_error_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Generate an error report for recent optimization failures.
        
        Args:
            hours_back: How many hours back to look for errors
            
        Returns:
            Comprehensive error report with statistics and details
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            # Get failed jobs from job manager
            failed_jobs = [
                job for job in job_manager.get_all_jobs()
                if job.get('status') == 'failed' and 
                datetime.fromisoformat(job.get('created_at', '1970-01-01')) > cutoff_time
            ]
            
            # Categorize errors
            error_categories = {}
            strategy_errors = {}
            
            for job in failed_jobs:
                error_msg = job.get('error_message', 'Unknown error')
                strategy = job.get('strategy_name', 'Unknown')
                
                # Categorize error
                if 'timeout' in error_msg.lower():
                    category = 'timeout_errors'
                elif 'parameter' in error_msg.lower() or 'validation' in error_msg.lower():
                    category = 'parameter_errors'
                elif 'memory' in error_msg.lower() or 'resource' in error_msg.lower():
                    category = 'resource_errors'
                elif 'numpy' in error_msg.lower() or 'integers' in error_msg.lower():
                    category = 'compatibility_errors'
                else:
                    category = 'general_errors'
                
                error_categories[category] = error_categories.get(category, 0) + 1
                
                # Track by strategy
                if strategy not in strategy_errors:
                    strategy_errors[strategy] = {'count': 0, 'errors': []}
                strategy_errors[strategy]['count'] += 1
                strategy_errors[strategy]['errors'].append({
                    'job_id': job.get('job_id'),
                    'error_message': error_msg,
                    'timestamp': job.get('created_at')
                })
            
            # Calculate error rate
            total_jobs = len(job_manager.get_all_jobs())
            total_failed = len(failed_jobs)
            error_rate = (total_failed / total_jobs * 100) if total_jobs > 0 else 0
            
            return {
                'report_period_hours': hours_back,
                'total_failed_jobs': total_failed,
                'total_jobs_in_period': total_jobs,
                'error_rate_percent': round(error_rate, 2),
                'error_categories': error_categories,
                'strategy_error_breakdown': strategy_errors,
                'recommendations': self._generate_error_recommendations(error_categories),
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate error report: {str(e)}")
            return {
                'error': 'Failed to generate error report',
                'message': str(e),
                'generated_at': datetime.utcnow().isoformat()
            }
    
    def _generate_error_recommendations(self, error_categories: Dict[str, int]) -> List[str]:
        """Generate recommendations based on error patterns."""
        recommendations = []
        
        if error_categories.get('timeout_errors', 0) > 5:
            recommendations.append("Consider increasing timeout values or reducing max_evals for optimization jobs")
        
        if error_categories.get('resource_errors', 0) > 3:
            recommendations.append("Monitor system resources and consider reducing parallel job count")
        
        if error_categories.get('compatibility_errors', 0) > 0:
            recommendations.append("Numpy compatibility issues detected - check library versions")
        
        if error_categories.get('parameter_errors', 0) > 2:
            recommendations.append("Review parameter validation logic and strategy configurations")
        
        if sum(error_categories.values()) > 10:
            recommendations.append("High error rate detected - consider system maintenance")
        
        return recommendations or ["No specific recommendations - error rate is within normal range"]

    def generate_batch_report(self, batch_id: str, format_type: str = "summary") -> Dict[str, Any]:
        """
        Generate a formatted report for a batch optimization result.
        
        Args:
            batch_id: ID of the batch to generate report for
            format_type: Type of report ('summary', 'detailed', 'csv', 'json')
            
        Returns:
            Dictionary containing the formatted report
        """
        try:
            # Get batch result from job manager
            batch_job = None
            for job in job_manager.get_all_jobs():
                if job.get('batch_id') == batch_id:
                    batch_job = job
                    break
            
            if not batch_job:
                raise ValueError(f"Batch {batch_id} not found")
            
            # Import result visualizer
            try:
                from src.optimization.result_visualizer import ResultVisualizer
            except ImportError:
                from ..optimization.result_visualizer import ResultVisualizer
            
            # For demo purposes, we'll create a mock batch result
            # In a real implementation, you'd retrieve the actual result
            from src.optimization.batch_manager import BatchOptimizationResult, BatchJobStatus
            from src.optimization.hyperopt_optimizer import OptimizationResult
            from datetime import datetime
            
            # Create mock successful job for demonstration
            mock_result = OptimizationResult(
                best_params={'param1': 0.5, 'param2': 10},
                best_score=1.85,
                best_metrics={
                    'sharpe_ratio': 1.85,
                    'annual_return': 0.45,
                    'max_drawdown': 0.125,
                    'total_trades': 150,
                    'win_rate': 0.68
                },
                all_trials=[],
                optimization_time=25.5,
                total_evaluations=100,
                cache_hits=0
            )
            
            mock_job = BatchJobStatus(
                strategy_name="MovingAverageCrossover",
                job_id=f"{batch_id}_job_0",
                status="completed",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                result=mock_result
            )
            
            mock_batch_result = BatchOptimizationResult(
                batch_id=batch_id,
                total_strategies=1,
                successful_jobs=[mock_job],
                failed_jobs=[],
                total_runtime_seconds=30.0,
                success_rate=100.0,
                summary_metrics={'note': 'Mock result for demonstration'},
                resource_usage={'cpu_percent': 25.5, 'memory_gb': 4.2},
                analysis_report={
                    'insights': {
                        'best_strategy': 'MovingAverageCrossover',
                        'performance_spread': 'Low variance',
                        'optimization_efficiency': 'Good'
                    }
                }
            )
            
            # Generate report
            visualizer = ResultVisualizer()
            
            if format_type == "summary":
                report_content = visualizer.generate_summary_report(mock_batch_result)
                content_type = "text/plain"
            elif format_type == "detailed":
                report_content = visualizer.generate_detailed_report(mock_batch_result)
                content_type = "text/plain"
            elif format_type == "csv":
                report_content = visualizer.generate_csv_export(mock_batch_result)
                content_type = "text/csv"
            elif format_type == "json":
                report_content = visualizer.generate_json_export(mock_batch_result)
                content_type = "application/json"
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
            
            return {
                "batch_id": batch_id,
                "format_type": format_type,
                "content_type": content_type,
                "report_content": report_content,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate batch report: {e}")
            raise ValueError(f"Report generation failed: {str(e)}")

    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """
        Get list of available strategies with their information.
        
        Returns:
            List of strategy information dictionaries
        """
        strategies = []
        
        for strategy_name, module_path in AVAILABLE_STRATEGIES.items():
            try:
                # For now, return basic info - we can enhance this later
                strategies.append({
                    "name": strategy_name,
                    "module": module_path,
                    "description": f"Trading strategy: {strategy_name}",
                    "parameters": [],  # TODO: Extract from strategy class
                    "category": self._get_strategy_category(strategy_name)
                })
            except Exception as e:
                logger.warning(f"Could not load strategy {strategy_name}: {e}")
                
        return strategies
    
    def _get_strategy_category(self, strategy_name: str) -> str:
        """Get strategy category based on name."""
        if "MA" in strategy_name or "Moving" in strategy_name:
            return "trend_following"
        elif "RSI" in strategy_name or "Stochastic" in strategy_name:
            return "momentum"
        elif "Bollinger" in strategy_name or "Keltner" in strategy_name:
            return "volatility"
        elif "Volume" in strategy_name or "OBV" in strategy_name or "VWAP" in strategy_name:
            return "volume"
        elif "MTF" in strategy_name:
            return "multi_timeframe"
        elif "Support" in strategy_name or "Fibonacci" in strategy_name:
            return "pattern_recognition"
        else:
            return "other"


# Global optimization service instance
optimization_service = OptimizationService() 
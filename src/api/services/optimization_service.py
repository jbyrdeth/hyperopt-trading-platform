"""
Optimization Service

Core business logic for managing optimization requests and coordinating
with the hyperparameter optimization engine and job management system.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio

from ..models import (
    OptimizationRequest, OptimizationResult, OptimizationStatus, 
    BatchOptimizationRequest, JobInfo, PerformanceMetrics, Asset, TimeFrame
)
from ..job_manager import job_manager, JobPriority
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
            
        except Exception as e:
            logger.error(f"Failed to submit optimization job: {str(e)}")
            raise ValueError(f"Failed to submit optimization: {str(e)}")
    
    async def submit_batch_optimization(self, request: BatchOptimizationRequest) -> Dict[str, Any]:
        """
        Submit multiple strategy optimization jobs.
        
        Args:
            request: Batch optimization request
            
        Returns:
            Batch job submission response with job_ids and status
        """
        try:
            submitted_jobs = []
            failed_submissions = []
            
            # Submit each strategy optimization
            for strategy_name in request.strategies:
                try:
                    # Create individual optimization request
                    individual_request = OptimizationRequest(
                        strategy_name=strategy_name,
                        asset=request.common_config.asset,
                        timeframe=request.common_config.timeframe,
                        start_date=request.common_config.start_date,
                        end_date=request.common_config.end_date,
                        optimization_config=request.strategy_specific_configs.get(
                            strategy_name, request.common_config.optimization_config
                        ),
                        validation_config=request.common_config.validation_config
                    )
                    
                    # Submit job with adjusted priority for batch processing
                    job_id = await job_manager.submit_job(individual_request, JobPriority.LOW)
                    
                    submitted_jobs.append({
                        "strategy_name": strategy_name,
                        "job_id": job_id,
                        "status": "queued"
                    })
                    
                except Exception as e:
                    failed_submissions.append({
                        "strategy_name": strategy_name,
                        "error": str(e)
                    })
            
            logger.info(f"Submitted batch optimization: {len(submitted_jobs)} jobs, {len(failed_submissions)} failures")
            
            return {
                "batch_id": f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "total_strategies": len(request.strategies),
                "submitted_jobs": submitted_jobs,
                "failed_submissions": failed_submissions,
                "success_rate": len(submitted_jobs) / len(request.strategies) * 100,
                "estimated_completion": self._estimate_batch_completion(len(submitted_jobs))
            }
            
        except Exception as e:
            logger.error(f"Failed to submit batch optimization: {str(e)}")
            raise ValueError(f"Failed to submit batch optimization: {str(e)}")
    
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
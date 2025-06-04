"""
Optimization Router

Handles optimization-related endpoints including job submission, status checking,
and result retrieval for the trading strategy optimization API.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging

from models import (
    OptimizationRequest, OptimizationResult, BatchOptimizationRequest,
    OptimizationStatus, BaseResponse, JobInfo, JobListResponse
)
from auth import verify_api_key, require_permission
from services.optimization_service import optimization_service
from job_manager import job_manager

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/single", response_model=Dict[str, Any])
async def submit_single_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    key_info: Dict = Depends(require_permission("write"))
):
    """
    Submit a single strategy optimization job.
    
    Starts an asynchronous optimization process for the specified strategy with
    the given parameters. Returns a job ID that can be used to track progress
    and retrieve results.
    
    **Features:**
    - Hyperparameter optimization using TPE algorithm
    - Real-time progress tracking
    - Comprehensive performance metrics
    - Out-of-sample validation
    - Statistical significance testing
    
    **Process:**
    1. Validates strategy and parameters
    2. Fetches historical market data
    3. Runs hyperparameter optimization
    4. Performs backtesting with best parameters
    5. Calculates performance metrics
    6. Generates validation results
    """
    try:
        logger.info(f"Submitting single optimization for {request.strategy_name}")
        
        # Submit optimization job
        result = await optimization_service.submit_single_optimization(request)
        
        # Schedule cleanup task
        background_tasks.add_task(job_manager.cleanup_old_jobs, max_age_hours=24)
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "OPTIMIZATION_VALIDATION_FAILED",
                "error_message": str(e),
                "request_details": {
                    "strategy": request.strategy_name,
                    "asset": request.asset.value,
                    "timeframe": request.timeframe.value
                }
            }
        )
    except Exception as e:
        logger.error(f"Failed to submit optimization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit optimization: {str(e)}"
        )


@router.post("/batch", response_model=Dict[str, Any])
async def submit_batch_optimization(
    request: BatchOptimizationRequest,
    background_tasks: BackgroundTasks,
    key_info: Dict = Depends(require_permission("write"))
):
    """
    Submit multiple strategy optimization jobs in a batch.
    
    Efficiently processes multiple strategies with shared configuration,
    allowing for comprehensive strategy comparison and analysis.
    
    **Batch Features:**
    - Parallel processing of multiple strategies
    - Shared configuration with strategy-specific overrides
    - Bulk progress tracking
    - Comparative performance analysis
    - Resource management and prioritization
    
    **Use Cases:**
    - Strategy comparison and ranking
    - Portfolio optimization
    - Market regime analysis across strategies
    - Bulk parameter sensitivity analysis
    """
    try:
        logger.info(f"Submitting batch optimization for {len(request.strategies)} strategies")
        
        # Validate batch size
        if len(request.strategies) > 20:
            raise HTTPException(
                status_code=422,
                detail="Batch size cannot exceed 20 strategies"
            )
        
        # Submit batch optimization
        result = await optimization_service.submit_batch_optimization(request)
        
        # Schedule cleanup task
        background_tasks.add_task(job_manager.cleanup_old_jobs, max_age_hours=24)
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "BATCH_OPTIMIZATION_VALIDATION_FAILED",
                "error_message": str(e),
                "batch_details": {
                    "strategy_count": len(request.strategies),
                    "strategies": request.strategies[:5]  # Show first 5
                }
            }
        )
    except Exception as e:
        logger.error(f"Failed to submit batch optimization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit batch optimization: {str(e)}"
        )


@router.get("/status/{job_id}", response_model=Dict[str, Any])
async def get_optimization_status(
    job_id: str,
    key_info: Dict = Depends(verify_api_key)
):
    """
    Get the current status of an optimization job.
    
    Provides real-time information about job progress, estimated completion time,
    current optimization phase, and system resource utilization.
    
    **Status Information:**
    - Current progress percentage
    - Optimization phase (data preparation, hyperparameter optimization, etc.)
    - Estimated completion time
    - Queue position (if queued)
    - Error information (if failed)
    - System load and resource utilization
    """
    try:
        status = optimization_service.get_optimization_status(job_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "JOB_NOT_FOUND",
                    "error_message": f"Optimization job '{job_id}' not found",
                    "job_id": job_id
                }
            )
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization status for {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve optimization status: {str(e)}"
        )


@router.get("/results/{job_id}", response_model=OptimizationResult)
async def get_optimization_results(
    job_id: str,
    key_info: Dict = Depends(verify_api_key)
):
    """
    Get the results of a completed optimization job.
    
    Returns comprehensive optimization results including best parameters,
    performance metrics, validation results, and detailed analysis.
    
    **Result Components:**
    - **Best Parameters:** Optimized hyperparameters for the strategy
    - **Performance Metrics:** Sharpe ratio, returns, drawdown, win rate, etc.
    - **Validation Results:** Out-of-sample testing, cross-asset validation
    - **Statistical Analysis:** Significance tests, confidence intervals
    - **Risk Metrics:** VaR, CVaR, maximum drawdown, volatility analysis
    """
    try:
        result = optimization_service.get_optimization_result(job_id)
        
        if not result:
            # Check if job exists but isn't completed
            status = optimization_service.get_optimization_status(job_id)
            if status:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error_code": "OPTIMIZATION_NOT_COMPLETED",
                        "error_message": f"Optimization job '{job_id}' is not yet completed",
                        "current_status": status["status"],
                        "progress": status["progress"]
                    }
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error_code": "JOB_NOT_FOUND",
                        "error_message": f"Optimization job '{job_id}' not found",
                        "job_id": job_id
                    }
                )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization results for {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve optimization results: {str(e)}"
        )


@router.delete("/cancel/{job_id}", response_model=BaseResponse)
async def cancel_optimization(
    job_id: str,
    key_info: Dict = Depends(require_permission("write"))
):
    """
    Cancel a queued or running optimization job.
    
    Gracefully stops the optimization process and cleans up resources.
    Only queued and running jobs can be cancelled.
    
    **Cancellation Process:**
    - Immediately stops hyperparameter optimization
    - Cleans up temporary files and resources
    - Updates job status to 'cancelled'
    - Preserves partial results if available
    """
    try:
        success = optimization_service.cancel_optimization(job_id)
        
        if not success:
            # Check if job exists
            status = optimization_service.get_optimization_status(job_id)
            if not status:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error_code": "JOB_NOT_FOUND",
                        "error_message": f"Optimization job '{job_id}' not found"
                    }
                )
            else:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error_code": "CANCELLATION_NOT_ALLOWED",
                        "error_message": f"Job '{job_id}' cannot be cancelled in status '{status['status']}'",
                        "current_status": status["status"]
                    }
                )
        
        logger.info(f"Optimization job {job_id} cancelled successfully")
        
        return BaseResponse(
            success=True,
            timestamp=None,
            request_id=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel optimization {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel optimization: {str(e)}"
        )


@router.get("/jobs", response_model=JobListResponse)
async def list_optimization_jobs(
    status: Optional[OptimizationStatus] = Query(None, description="Filter by job status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of jobs to return"),
    key_info: Dict = Depends(verify_api_key)
):
    """
    List optimization jobs with optional filtering.
    
    Returns a paginated list of optimization jobs with comprehensive information
    including status, progress, creation time, and performance summary.
    
    **Filtering Options:**
    - **Status:** Filter by job status (queued, running, completed, failed, cancelled)
    - **Limit:** Control the number of results returned
    
    **Job Information:**
    - Job ID and creation timestamp
    - Strategy name and configuration
    - Current status and progress
    - Estimated completion time
    - Performance summary (for completed jobs)
    """
    try:
        jobs = optimization_service.list_optimization_jobs(status, limit)
        
        # Count jobs by status
        all_jobs = optimization_service.list_optimization_jobs(None, 1000)  # Get all for counting
        status_counts = {}
        for job in all_jobs:
            job_status = job["status"]
            status_counts[job_status] = status_counts.get(job_status, 0) + 1
        
        # Convert to JobInfo models
        job_infos = []
        for job in jobs:
            job_info = JobInfo(
                job_id=job["job_id"],
                job_type="optimization",
                status=OptimizationStatus(job["status"]),
                priority=job.get("priority", 2),
                created_at=job["created_at"],
                started_at=job.get("started_at"),
                estimated_completion=job.get("estimated_completion"),
                progress=job.get("progress")
            )
            job_infos.append(job_info)
        
        return JobListResponse(
            jobs=job_infos,
            total_count=len(all_jobs),
            active_count=status_counts.get("running", 0),
            queued_count=status_counts.get("queued", 0)
        )
        
    except Exception as e:
        logger.error(f"Failed to list optimization jobs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list optimization jobs: {str(e)}"
        )


@router.get("/statistics", response_model=Dict[str, Any])
async def get_optimization_statistics(
    key_info: Dict = Depends(verify_api_key)
):
    """
    Get comprehensive optimization system statistics.
    
    Provides insights into system performance, resource utilization,
    job throughput, and overall optimization effectiveness.
    
    **Statistics Categories:**
    - **Queue Statistics:** Active jobs, queue size, completion rates
    - **Performance Metrics:** Average completion times, success rates, throughput
    - **Resource Utilization:** CPU, memory, and worker utilization
    - **System Capacity:** Maximum concurrent jobs, queue limits, current load
    
    **Use Cases:**
    - System monitoring and alerting
    - Capacity planning
    - Performance optimization
    - Resource allocation decisions
    """
    try:
        stats = optimization_service.get_optimization_statistics()
        
        # Add additional system information
        stats["api_info"] = {
            "supported_strategies": 65,  # TODO: Get from strategy factory
            "supported_assets": len([asset.value for asset in [
                "BTC", "ETH", "BNB", "ADA", "SOL", "DOT", "LINK", "MATIC"
            ]]),
            "supported_timeframes": len([tf.value for tf in [
                "1m", "5m", "15m", "30m", "1h", "4h", "8h", "1d", "1w"
            ]]),
            "optimization_algorithms": ["tpe", "random"],
            "validation_methods": [
                "out_of_sample", "cross_asset", "monte_carlo", "statistical_tests"
            ]
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get optimization statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve optimization statistics: {str(e)}"
        )


@router.post("/validate", response_model=BaseResponse)
async def validate_optimization_request(
    request: OptimizationRequest,
    key_info: Dict = Depends(verify_api_key)
):
    """
    Validate an optimization request without submitting it.
    
    Performs comprehensive validation of the optimization configuration
    including strategy existence, parameter ranges, data availability,
    and resource requirements.
    
    **Validation Checks:**
    - **Strategy Validation:** Verify strategy exists and is available
    - **Parameter Validation:** Check parameter types and ranges
    - **Data Validation:** Confirm data availability for the specified period
    - **Resource Validation:** Estimate resource requirements and availability
    - **Configuration Validation:** Verify optimization settings are valid
    
    **Use Cases:**
    - Pre-submission validation
    - Configuration testing
    - Resource planning
    - Error prevention
    """
    try:
        # Use the validation logic from the optimization service
        await optimization_service._validate_optimization_request(request)
        
        return BaseResponse(
            success=True,
            timestamp=None,
            request_id=None
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "VALIDATION_FAILED",
                "error_message": str(e),
                "validation_details": {
                    "strategy": request.strategy_name,
                    "asset": request.asset.value,
                    "timeframe": request.timeframe.value,
                    "date_range_days": (request.end_date - request.start_date).days
                }
            }
        )
    except Exception as e:
        logger.error(f"Failed to validate optimization request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate optimization request: {str(e)}"
        ) 
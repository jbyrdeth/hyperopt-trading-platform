"""
Job Manager for Background Processing

Handles asynchronous optimization jobs with progress tracking, status updates,
and result storage for the trading strategy optimization API.
"""

import asyncio
import logging
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Callable
from enum import Enum
import json
import os
from concurrent.futures import ThreadPoolExecutor
import psutil

from models import OptimizationRequest, OptimizationResult, OptimizationStatus, PerformanceMetrics

logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


class OptimizationJob:
    """Individual optimization job representation."""
    
    def __init__(self, job_id: str, request: OptimizationRequest, priority: JobPriority = JobPriority.NORMAL):
        self.job_id = job_id
        self.request = request
        self.priority = priority
        self.status = OptimizationStatus.QUEUED
        self.progress = 0.0
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.result: Optional[OptimizationResult] = None
        self.task: Optional[asyncio.Task] = None
        self.estimated_completion: Optional[datetime] = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for API responses."""
        return {
            "job_id": self.job_id,
            "strategy_name": self.request.strategy_name,
            "status": self.status.value,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "error_message": self.error_message,
            "asset": self.request.asset.value,
            "timeframe": self.request.timeframe.value,
            "priority": self.priority.value
        }
        
    def get_duration(self) -> Optional[float]:
        """Get job duration in seconds if completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class OptimizationJobManager:
    """
    Manages background optimization jobs with queue processing, status tracking,
    and resource management.
    """
    
    def __init__(self, max_concurrent_jobs: int = 3, max_queue_size: int = 100, metrics_collector=None):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_queue_size = max_queue_size
        self.metrics_collector = metrics_collector
        
        # Job storage
        self.jobs: Dict[str, OptimizationJob] = {}
        self.job_queue: Optional[asyncio.PriorityQueue] = None
        self.active_jobs: Dict[str, OptimizationJob] = {}
        self.completed_jobs: Dict[str, OptimizationJob] = {}
        
        # Worker management
        self.workers: list = []
        self.worker_executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self.shutdown_event: Optional[asyncio.Event] = None
        self.initialized = False
        
        # Statistics
        self.total_jobs_processed = 0
        self.total_jobs_failed = 0
        self.average_completion_time = 0.0
        
        logger.info(f"OptimizationJobManager created with {max_concurrent_jobs} workers (will initialize on first use)")
    
    def _record_metrics(self):
        """Record current metrics if collector is available."""
        if self.metrics_collector:
            # Record queue sizes by priority
            queue_stats = self.get_queue_stats()
            for priority, count in queue_stats.get("by_priority", {}).items():
                self.metrics_collector.record_queue_size(priority, count)
            
            # Record worker metrics
            self.metrics_collector.record_worker_metrics(
                pool_size=self.max_concurrent_jobs,
                busy_count=len(self.active_jobs)
            )
    
    async def _ensure_initialized(self):
        """Ensure the job manager is initialized with async components."""
        if not self.initialized:
            self.job_queue = asyncio.PriorityQueue(maxsize=self.max_queue_size)
            self.shutdown_event = asyncio.Event()
            self._start_workers()
            self.initialized = True
            logger.info("OptimizationJobManager initialized with async components")
    
    def _start_workers(self):
        """Start background worker tasks."""
        for i in range(self.max_concurrent_jobs):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
    
    async def _worker(self, worker_name: str):
        """
        Background worker that processes optimization jobs from the queue.
        """
        logger.info(f"Worker {worker_name} started")
        
        while not (self.shutdown_event and self.shutdown_event.is_set()):
            try:
                # Record current metrics
                self._record_metrics()
                
                # Get next job from queue (blocks until available)
                priority, job_id = await asyncio.wait_for(
                    self.job_queue.get(), 
                    timeout=1.0
                )
                
                job = self.jobs[job_id]
                
                # Record job started
                if self.metrics_collector:
                    self.metrics_collector.record_job_started(job_id, job.request.strategy_name)
                
                # Move job to active state
                self.active_jobs[job_id] = job
                job.status = OptimizationStatus.RUNNING
                job.started_at = datetime.utcnow()
                
                logger.info(f"Worker {worker_name} starting job {job_id}")
                
                # Estimate completion time based on historical data
                job.estimated_completion = self._estimate_completion_time(job)
                
                # Process the optimization
                await self._process_optimization_job(job)
                
                # Move to completed jobs
                job.completed_at = datetime.utcnow()
                self.completed_jobs[job_id] = job
                del self.active_jobs[job_id]
                
                # Record job completion
                if self.metrics_collector:
                    duration = job.get_duration() or 0.0
                    status = job.status.value
                    self.metrics_collector.record_job_completed(job_id, job.request.strategy_name, duration, status)
                
                # Update statistics
                self._update_statistics(job)
                
                logger.info(f"Worker {worker_name} completed job {job_id} with status {job.status}")
                
            except asyncio.TimeoutError:
                # No jobs available, continue waiting
                continue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {str(e)}")
                if 'job' in locals() and job:
                    job.status = OptimizationStatus.FAILED
                    job.error_message = str(e)
                    job.completed_at = datetime.utcnow()
                    self.completed_jobs[job_id] = job
                    if job_id in self.active_jobs:
                        del self.active_jobs[job_id]
                    
                    # Record failed job
                    if self.metrics_collector:
                        duration = job.get_duration() or 0.0
                        self.metrics_collector.record_job_completed(job_id, job.request.strategy_name, duration, "failed")
    
    async def submit_job(self, request: OptimizationRequest, priority: JobPriority = JobPriority.NORMAL) -> str:
        """
        Submit a new optimization job to the queue.
        
        Args:
            request: Optimization request configuration
            priority: Job priority level
            
        Returns:
            Unique job ID
            
        Raises:
            ValueError: If queue is full
        """
        await self._ensure_initialized()
        
        if self.job_queue.qsize() >= self.max_queue_size:
            raise ValueError("Job queue is full. Please try again later.")
        
        # Generate unique job ID
        job_id = f"opt_{uuid.uuid4().hex[:8]}"
        
        # Create job
        job = OptimizationJob(job_id, request, priority)
        self.jobs[job_id] = job
        
        # Record job submission
        if self.metrics_collector:
            self.metrics_collector.record_job_submitted(request.strategy_name)
        
        # Add to queue (lower priority value = higher priority)
        await self.job_queue.put((priority.value, job_id))
        
        logger.info(f"Job {job_id} submitted for strategy {request.strategy_name}")
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a job."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        return job.to_dict()
    
    def get_job_result(self, job_id: str) -> Optional[OptimizationResult]:
        """Get optimization result for a completed job."""
        job = self.jobs.get(job_id)
        if not job or job.status != OptimizationStatus.COMPLETED:
            return None
        
        return job.result
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a queued or running job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if job was cancelled, False if not found or already completed
        """
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [OptimizationStatus.COMPLETED, OptimizationStatus.FAILED]:
            return False
        
        job.status = OptimizationStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        
        # Cancel task if running
        if job.task and not job.task.done():
            job.task.cancel()
        
        # Remove from active jobs
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
        
        logger.info(f"Job {job_id} cancelled")
        return True
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get job queue statistics."""
        return {
            "queue_size": self.job_queue.qsize() if self.job_queue else 0,
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "total_jobs": len(self.jobs),
            "total_processed": self.total_jobs_processed,
            "total_failed": self.total_jobs_failed,
            "success_rate": self._calculate_success_rate(),
            "average_completion_time_minutes": self.average_completion_time,
            "workers_active": len([w for w in self.workers if not w.done()]),
            "system_load": self._get_system_load()
        }
    
    def list_jobs(self, status: Optional[OptimizationStatus] = None, limit: int = 50) -> list:
        """List jobs with optional status filtering."""
        jobs = list(self.jobs.values())
        
        if status:
            jobs = [job for job in jobs if job.status == status]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        return [job.to_dict() for job in jobs[:limit]]
    
    async def _process_optimization_job(self, job: OptimizationJob):
        """
        Process a single optimization job.
        
        This is where we'll integrate with the actual optimization engine.
        """
        try:
            # Progress callback for real-time updates
            def progress_callback(progress: float, message: str = ""):
                job.progress = min(100.0, max(0.0, progress))
                
                # Record progress metrics
                if self.metrics_collector:
                    self.metrics_collector.record_job_progress(job.job_id, job.request.strategy_name, progress)
                
                logger.debug(f"Job {job.job_id} progress: {progress:.1f}% - {message}")
            
            # TODO: Integrate with actual optimization engine
            # For now, simulate optimization process
            await self._simulate_optimization(job, progress_callback)
            
            job.status = OptimizationStatus.COMPLETED
            
            # Record strategy performance if available
            if job.result and self.metrics_collector:
                metrics = job.result.performance_metrics
                self.metrics_collector.record_strategy_performance(
                    job.request.strategy_name, "sharpe_ratio", metrics.sharpe_ratio
                )
                self.metrics_collector.record_strategy_performance(
                    job.request.strategy_name, "total_return", metrics.total_return
                )
                self.metrics_collector.record_strategy_performance(
                    job.request.strategy_name, "max_drawdown", metrics.max_drawdown
                )
            
        except asyncio.CancelledError:
            job.status = OptimizationStatus.CANCELLED
            logger.info(f"Job {job.job_id} was cancelled")
        except Exception as e:
            job.status = OptimizationStatus.FAILED
            job.error_message = str(e)
            self.total_jobs_failed += 1
            logger.error(f"Job {job.job_id} failed: {str(e)}")
    
    async def _simulate_optimization(self, job: OptimizationJob, progress_callback: Callable):
        """
        Simulate optimization process (placeholder for actual integration).
        
        This will be replaced with actual optimization engine integration.
        """
        request = job.request
        
        # Simulate optimization steps
        steps = [
            (10, "Validating strategy parameters"),
            (20, "Fetching historical data"),
            (40, "Running hyperparameter optimization"),
            (70, "Backtesting best parameters"),
            (85, "Calculating performance metrics"),
            (95, "Generating validation results"),
            (100, "Optimization completed")
        ]
        
        for progress, message in steps:
            progress_callback(progress, message)
            
            # Simulate processing time
            await asyncio.sleep(0.5)
            
            # Check for cancellation
            if job.status == OptimizationStatus.CANCELLED:
                raise asyncio.CancelledError()
        
        # Create mock result
        job.result = OptimizationResult(
            job_id=job.job_id,
            strategy_name=request.strategy_name,
            status=OptimizationStatus.COMPLETED,
            progress=100.0,
            best_parameters={
                "fast_period": 12,
                "slow_period": 26,
                "signal_threshold": 0.02
            },
            best_score=1.85,
            performance_metrics=PerformanceMetrics(
                total_return=45.2,
                sharpe_ratio=1.85,
                sortino_ratio=2.1,
                calmar_ratio=1.3,
                max_drawdown=12.5,
                volatility=18.7,
                win_rate=0.68,
                profit_factor=1.75,
                trades_count=156,
                avg_trade_return=0.29
            ),
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=datetime.utcnow()
        )
    
    def _estimate_completion_time(self, job: OptimizationJob) -> datetime:
        """Estimate job completion time based on configuration and historical data."""
        base_time = job.request.optimization_config.timeout_minutes
        
        # Adjust based on strategy complexity and evaluation count
        complexity_factor = 1.0  # TODO: Get from strategy metadata
        eval_factor = job.request.optimization_config.max_evals / 100.0
        
        estimated_minutes = base_time * complexity_factor * eval_factor
        
        return datetime.utcnow() + timedelta(minutes=estimated_minutes)
    
    def _update_statistics(self, job: OptimizationJob):
        """Update job processing statistics."""
        self.total_jobs_processed += 1
        
        if job.status == OptimizationStatus.COMPLETED and job.started_at and job.completed_at:
            completion_time = (job.completed_at - job.started_at).total_seconds() / 60.0
            
            # Update rolling average
            if self.average_completion_time == 0:
                self.average_completion_time = completion_time
            else:
                self.average_completion_time = (
                    self.average_completion_time * 0.9 + completion_time * 0.1
                )
    
    def _calculate_success_rate(self) -> float:
        """Calculate job success rate."""
        if self.total_jobs_processed == 0:
            return 0.0
        
        success_count = self.total_jobs_processed - self.total_jobs_failed
        return (success_count / self.total_jobs_processed) * 100.0
    
    def _get_system_load(self) -> Dict[str, float]:
        """Get current system resource usage."""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "active_workers": len(self.active_jobs)
            }
        except Exception:
            return {"cpu_percent": 0, "memory_percent": 0, "active_workers": 0}
    
    async def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs to prevent memory buildup."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        jobs_to_remove = []
        for job_id, job in self.completed_jobs.items():
            if job.completed_at and job.completed_at < cutoff_time:
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.completed_jobs[job_id]
            del self.jobs[job_id]
        
        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
    
    async def shutdown(self):
        """Gracefully shutdown the job manager."""
        logger.info("Shutting down OptimizationJobManager...")
        
        # Signal workers to stop
        self.shutdown_event.set()
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Shutdown thread pool
        self.worker_executor.shutdown(wait=True)
        
        logger.info("OptimizationJobManager shutdown complete")


# Global job manager instance with metrics collection
def _create_job_manager():
    """Create job manager with metrics collector if available."""
    try:
        from monitoring import get_metrics_collector
        metrics_collector = get_metrics_collector()
        return OptimizationJobManager(metrics_collector=metrics_collector)
    except ImportError:
        # Fallback if monitoring is not available
        return OptimizationJobManager()

job_manager = _create_job_manager() 
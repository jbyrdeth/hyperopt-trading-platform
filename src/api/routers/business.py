from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pydantic import BaseModel
import pandas as pd
import numpy as np
from ..models import BusinessMetrics, ROICalculation, ExecutiveSummary, StrategyComparison
from ..services.business_service import BusinessService
from ..services.roi_calculator import ROICalculator
from ..auth import get_current_user, verify_premium_access
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/business", tags=["business"])

class ROIRequest(BaseModel):
    strategy_ids: List[str]
    calculation_method: str = "compound"  # compound, simple, annualized, sharpe_adjusted
    time_period_days: Optional[int] = None
    benchmark: Optional[str] = "spy"  # spy, nasdaq, custom
    include_fees: bool = True
    risk_free_rate: float = 0.03

class BusinessDashboardRequest(BaseModel):
    date_range: str = "30d"  # 7d, 30d, 90d, 1y, all
    strategy_filter: Optional[List[str]] = None
    metric_types: List[str] = ["roi", "sharpe", "max_drawdown", "win_rate", "total_trades"]
    include_forecasts: bool = False

class ReportExportRequest(BaseModel):
    report_type: str  # executive_summary, strategy_comparison, performance_analysis
    format: str = "pdf"  # pdf, excel, csv, json
    date_range: str = "30d"
    strategy_ids: Optional[List[str]] = None
    include_charts: bool = True

@router.get("/dashboard/executive", response_model=Dict[str, Any])
async def get_executive_dashboard(
    date_range: str = Query("30d", description="Time range: 7d, 30d, 90d, 1y, all"),
    current_user = Depends(get_current_user)
):
    """
    Get executive dashboard with key business metrics and KPIs.
    Designed for C-level executives and business stakeholders.
    """
    try:
        business_service = BusinessService()
        dashboard_data = await business_service.get_executive_dashboard(
            date_range=date_range,
            user_id=current_user.id
        )
        
        return {
            "success": True,
            "data": dashboard_data,
            "generated_at": datetime.utcnow().isoformat(),
            "user_role": current_user.role
        }
    except Exception as e:
        logger.error(f"Executive dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate executive dashboard: {str(e)}")

@router.post("/roi/calculate", response_model=Dict[str, Any])
async def calculate_roi(
    request: ROIRequest,
    current_user = Depends(get_current_user)
):
    """
    Calculate ROI for selected strategies using various methodologies.
    Supports compound, simple, annualized, and Sharpe-adjusted calculations.
    """
    try:
        roi_calculator = ROICalculator()
        roi_results = await roi_calculator.calculate_comprehensive_roi(
            strategy_ids=request.strategy_ids,
            calculation_method=request.calculation_method,
            time_period_days=request.time_period_days,
            benchmark=request.benchmark,
            include_fees=request.include_fees,
            risk_free_rate=request.risk_free_rate,
            user_id=current_user.id
        )
        
        return {
            "success": True,
            "calculation_method": request.calculation_method,
            "benchmark": request.benchmark,
            "results": roi_results,
            "calculated_at": datetime.utcnow().isoformat(),
            "metadata": {
                "strategies_analyzed": len(request.strategy_ids),
                "time_period_days": request.time_period_days,
                "includes_fees": request.include_fees
            }
        }
    except Exception as e:
        logger.error(f"ROI calculation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ROI calculation failed: {str(e)}")

@router.get("/metrics/realtime", response_model=Dict[str, Any])
async def get_realtime_business_metrics(
    current_user = Depends(get_current_user)
):
    """
    Get real-time business metrics including active optimizations,
    current P&L, platform usage, and performance indicators.
    """
    try:
        business_service = BusinessService()
        realtime_metrics = await business_service.get_realtime_metrics(
            user_id=current_user.id
        )
        
        return {
            "success": True,
            "metrics": realtime_metrics,
            "timestamp": datetime.utcnow().isoformat(),
            "refresh_interval_seconds": 30
        }
    except Exception as e:
        logger.error(f"Realtime metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get realtime metrics: {str(e)}")

@router.post("/strategies/compare", response_model=Dict[str, Any])
async def compare_strategies(
    strategy_ids: List[str],
    metrics: List[str] = Query(["roi", "sharpe", "max_drawdown", "volatility"]),
    date_range: str = Query("90d"),
    current_user = Depends(get_current_user)
):
    """
    Compare multiple strategies across key business metrics.
    Provides side-by-side analysis for decision making.
    """
    try:
        business_service = BusinessService()
        comparison_data = await business_service.compare_strategies(
            strategy_ids=strategy_ids,
            metrics=metrics,
            date_range=date_range,
            user_id=current_user.id
        )
        
        return {
            "success": True,
            "comparison": comparison_data,
            "strategies_compared": len(strategy_ids),
            "metrics_analyzed": metrics,
            "date_range": date_range,
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Strategy comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Strategy comparison failed: {str(e)}")

@router.post("/reports/export", response_model=Dict[str, Any])
async def export_business_report(
    request: ReportExportRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """
    Export business reports in various formats (PDF, Excel, CSV).
    Generates comprehensive reports for sharing with stakeholders.
    """
    try:
        business_service = BusinessService()
        
        # Generate report asynchronously for large datasets
        report_id = await business_service.queue_report_generation(
            report_type=request.report_type,
            format=request.format,
            date_range=request.date_range,
            strategy_ids=request.strategy_ids,
            include_charts=request.include_charts,
            user_id=current_user.id
        )
        
        return {
            "success": True,
            "report_id": report_id,
            "status": "queued",
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
            "download_url": f"/api/business/reports/download/{report_id}",
            "format": request.format
        }
    except Exception as e:
        logger.error(f"Report export error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report export failed: {str(e)}")

@router.get("/reports/download/{report_id}")
async def download_report(
    report_id: str,
    current_user = Depends(get_current_user)
):
    """
    Download generated business report by ID.
    """
    try:
        business_service = BusinessService()
        report_path = await business_service.get_report_path(
            report_id=report_id,
            user_id=current_user.id
        )
        
        if not report_path:
            raise HTTPException(status_code=404, detail="Report not found or not ready")
            
        return FileResponse(
            path=report_path,
            filename=f"hyperopt_business_report_{report_id}.{report_path.split('.')[-1]}",
            media_type="application/octet-stream"
        )
    except Exception as e:
        logger.error(f"Report download error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report download failed: {str(e)}")

@router.get("/analytics/trends", response_model=Dict[str, Any])
async def get_business_trends(
    metric: str = Query("roi", description="Metric to trend: roi, trades, profit, drawdown"),
    period: str = Query("daily", description="Trend period: hourly, daily, weekly, monthly"),
    duration: str = Query("30d", description="Duration: 7d, 30d, 90d, 1y"),
    current_user = Depends(get_current_user)
):
    """
    Get business trend analysis for key metrics over time.
    Provides insights into platform performance evolution.
    """
    try:
        business_service = BusinessService()
        trend_data = await business_service.get_trend_analysis(
            metric=metric,
            period=period,
            duration=duration,
            user_id=current_user.id
        )
        
        return {
            "success": True,
            "metric": metric,
            "period": period,
            "duration": duration,
            "trend_data": trend_data,
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Trend analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")

@router.get("/portfolio/summary", response_model=Dict[str, Any])
async def get_portfolio_summary(
    include_inactive: bool = Query(False),
    current_user = Depends(get_current_user)
):
    """
    Get comprehensive portfolio summary for business reporting.
    Includes allocation, performance, and risk metrics.
    """
    try:
        business_service = BusinessService()
        portfolio_summary = await business_service.get_portfolio_summary(
            user_id=current_user.id,
            include_inactive=include_inactive
        )
        
        return {
            "success": True,
            "portfolio": portfolio_summary,
            "generated_at": datetime.utcnow().isoformat(),
            "includes_inactive": include_inactive
        }
    except Exception as e:
        logger.error(f"Portfolio summary error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Portfolio summary failed: {str(e)}")

@router.post("/alerts/setup", response_model=Dict[str, Any])
async def setup_business_alerts(
    metric: str,
    threshold: float,
    condition: str,  # above, below, equals
    notification_method: str = "email",  # email, slack, webhook
    current_user = Depends(verify_premium_access)
):
    """
    Set up business metric alerts for proactive monitoring.
    Premium feature for advanced users.
    """
    try:
        business_service = BusinessService()
        alert_id = await business_service.create_business_alert(
            metric=metric,
            threshold=threshold,
            condition=condition,
            notification_method=notification_method,
            user_id=current_user.id
        )
        
        return {
            "success": True,
            "alert_id": alert_id,
            "metric": metric,
            "threshold": threshold,
            "condition": condition,
            "notification_method": notification_method,
            "created_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Alert setup error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Alert setup failed: {str(e)}")

@router.get("/performance/benchmark", response_model=Dict[str, Any])
async def get_benchmark_comparison(
    benchmark: str = Query("spy", description="Benchmark: spy, nasdaq, custom"),
    strategies: Optional[List[str]] = Query(None),
    date_range: str = Query("90d"),
    current_user = Depends(get_current_user)
):
    """
    Compare platform performance against market benchmarks.
    Essential for business performance evaluation.
    """
    try:
        business_service = BusinessService()
        benchmark_data = await business_service.get_benchmark_comparison(
            benchmark=benchmark,
            strategies=strategies,
            date_range=date_range,
            user_id=current_user.id
        )
        
        return {
            "success": True,
            "benchmark": benchmark,
            "comparison_data": benchmark_data,
            "strategies_included": strategies or "all",
            "date_range": date_range,
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Benchmark comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Benchmark comparison failed: {str(e)}") 
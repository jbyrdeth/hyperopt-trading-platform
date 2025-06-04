"""
API Models

Comprehensive Pydantic models for request and response validation.
These models ensure type safety and automatic API documentation.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any, Literal
from datetime import datetime
from enum import Enum
import uuid


# Enumerations
class StrategyType(str, Enum):
    """Strategy category types."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion" 
    MOMENTUM = "momentum"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    PATTERN = "pattern"
    MULTI_TIMEFRAME = "multi_timeframe"


class OptimizationStatus(str, Enum):
    """Optimization job status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TimeFrame(str, Enum):
    """Available timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    H8 = "8h"
    D1 = "1d"
    W1 = "1w"


class Asset(str, Enum):
    """Supported trading assets."""
    BTC = "BTC"
    ETH = "ETH"
    BNB = "BNB"
    ADA = "ADA"
    SOL = "SOL"
    DOT = "DOT"
    LINK = "LINK"
    MATIC = "MATIC"


class ReportType(str, Enum):
    """Report generation types."""
    FULL = "full"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"


class ExportFormat(str, Enum):
    """Export file formats."""
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    PINE_SCRIPT = "pine_script"


# Base Models
class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = False
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None


# Strategy Models
class StrategyParameter(BaseModel):
    """Individual strategy parameter."""
    name: str
    value: Union[int, float, str, bool]
    type: str
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    description: Optional[str] = None


class StrategyInfo(BaseModel):
    """Strategy information model."""
    name: str
    description: str
    category: StrategyType
    parameters: Dict[str, StrategyParameter]
    default_timeframe: TimeFrame = TimeFrame.H4
    recommended_assets: List[Asset] = [Asset.BTC]
    risk_level: str = Field(..., description="Low, Medium, or High")
    complexity_score: Optional[float] = Field(None, ge=1, le=10)
    
    class Config:
        schema_extra = {
            "example": {
                "name": "MovingAverageCrossover",
                "description": "Moving average crossover strategy with signal filtering",
                "category": "trend_following",
                "parameters": {
                    "fast_period": {
                        "name": "fast_period",
                        "value": 12,
                        "type": "int",
                        "min_value": 1,
                        "max_value": 100,
                        "description": "Fast moving average period"
                    }
                },
                "risk_level": "Medium",
                "complexity_score": 6.5
            }
        }


class StrategyListResponse(BaseResponse):
    """Response for strategy list endpoint."""
    strategies: List[StrategyInfo]
    total_count: int
    categories: Dict[str, int] = Field(..., description="Count of strategies per category")


# Optimization Models
class OptimizationConfig(BaseModel):
    """Optimization configuration."""
    max_evals: int = Field(100, ge=10, le=1000, description="Maximum optimization iterations")
    timeout_minutes: int = Field(60, ge=5, le=480, description="Optimization timeout in minutes")
    n_startup_jobs: int = Field(10, ge=1, le=50, description="Number of random startup jobs")
    algorithm: str = Field("tpe", description="Optimization algorithm (tpe, random)")
    objective: str = Field("sharpe_ratio", description="Optimization objective")
    cross_validation: bool = Field(True, description="Enable cross-validation during optimization")
    
    class Config:
        schema_extra = {
            "example": {
                "max_evals": 200,
                "timeout_minutes": 120,
                "n_startup_jobs": 20,
                "algorithm": "tpe",
                "objective": "sharpe_ratio",
                "cross_validation": True
            }
        }


class ValidationConfig(BaseModel):
    """Validation configuration."""
    out_of_sample_ratio: float = Field(0.3, ge=0.1, le=0.5, description="Out-of-sample test ratio")
    monte_carlo_runs: int = Field(100, ge=10, le=1000, description="Monte Carlo simulation runs")
    cross_asset_validation: bool = Field(True, description="Enable cross-asset validation")
    regime_analysis: bool = Field(True, description="Enable market regime analysis")
    statistical_tests: bool = Field(True, description="Enable statistical significance tests")


class OptimizationRequest(BaseModel):
    """Optimization request model."""
    strategy_name: str = Field(..., description="Name of the strategy to optimize")
    asset: Asset = Field(Asset.BTC, description="Trading asset")
    timeframe: TimeFrame = Field(TimeFrame.H4, description="Trading timeframe")
    start_date: datetime = Field(..., description="Start date for historical data")
    end_date: datetime = Field(..., description="End date for historical data")
    optimization_config: OptimizationConfig = OptimizationConfig()
    validation_config: Optional[ValidationConfig] = ValidationConfig()
    custom_parameters: Optional[Dict[str, Any]] = Field(None, description="Custom parameter overrides")
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "strategy_name": "MovingAverageCrossover",
                "asset": "BTC",
                "timeframe": "4h",
                "start_date": "2023-01-01T00:00:00Z",
                "end_date": "2023-12-31T23:59:59Z",
                "optimization_config": {
                    "max_evals": 200,
                    "timeout_minutes": 120,
                    "objective": "sharpe_ratio"
                }
            }
        }


class PerformanceMetrics(BaseModel):
    """Performance metrics model."""
    total_return: float = Field(..., description="Total return percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    calmar_ratio: float = Field(..., description="Calmar ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    volatility: float = Field(..., description="Annualized volatility")
    win_rate: float = Field(..., ge=0, le=1, description="Win rate (0-1)")
    profit_factor: float = Field(..., description="Profit factor")
    trades_count: int = Field(..., ge=0, description="Total number of trades")
    avg_trade_return: float = Field(..., description="Average trade return")


class OptimizationResult(BaseModel):
    """Optimization result model."""
    job_id: str = Field(..., description="Unique job identifier")
    strategy_name: str
    status: OptimizationStatus
    progress: Optional[float] = Field(None, ge=0, le=100, description="Progress percentage")
    best_parameters: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    validation_results: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "opt_abc123",
                "strategy_name": "MovingAverageCrossover",
                "status": "completed",
                "progress": 100.0,
                "best_parameters": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_threshold": 0.02
                },
                "best_score": 1.85,
                "performance_metrics": {
                    "total_return": 45.2,
                    "sharpe_ratio": 1.85,
                    "max_drawdown": 12.5,
                    "win_rate": 0.68
                }
            }
        }


class BatchOptimizationRequest(BaseModel):
    """Batch optimization request model."""
    strategies: List[str] = Field(..., description="List of strategy names to optimize")
    common_config: OptimizationRequest = Field(..., description="Common configuration for all strategies")
    strategy_specific_configs: Optional[Dict[str, OptimizationConfig]] = Field(
        None, description="Strategy-specific optimization configurations"
    )
    parallel_jobs: int = Field(3, ge=1, le=10, description="Number of parallel optimization jobs")


# Validation Models
class ValidationRequest(BaseModel):
    """Validation request model."""
    optimization_result: OptimizationResult = Field(..., description="Optimization result to validate")
    validation_type: str = Field(..., description="Type of validation to perform")
    additional_assets: Optional[List[Asset]] = Field(None, description="Additional assets for cross-validation")
    config: Optional[ValidationConfig] = ValidationConfig()


class ValidationResult(BaseModel):
    """Validation result model."""
    validation_type: str
    success: bool
    score: float = Field(..., ge=0, le=10, description="Validation score (0-10)")
    details: Dict[str, Any]
    warnings: List[str] = []
    recommendations: List[str] = []


# Export Models
class PineScriptRequest(BaseModel):
    """Pine Script generation request."""
    strategy_name: str
    optimized_parameters: Dict[str, Any]
    template_type: str = Field("strategy", description="Pine Script type: strategy or indicator")
    include_alerts: bool = Field(True, description="Include alert conditions")
    include_debugging: bool = Field(True, description="Include debugging features")
    include_visualization: bool = Field(True, description="Include visualization plots")
    
    class Config:
        schema_extra = {
            "example": {
                "strategy_name": "MovingAverageCrossover",
                "optimized_parameters": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_threshold": 0.02
                },
                "template_type": "strategy",
                "include_alerts": True
            }
        }


class PineScriptResponse(BaseResponse):
    """Pine Script generation response."""
    script_content: str = Field(..., description="Generated Pine Script v5 code")
    script_name: str = Field(..., description="Generated script filename")
    validation_issues: List[str] = Field([], description="Script validation warnings")
    download_url: Optional[str] = Field(None, description="URL to download the script file")


class ReportRequest(BaseModel):
    """Report generation request."""
    optimization_results: List[OptimizationResult] = Field(..., description="Optimization results to include")
    report_type: ReportType = ReportType.FULL
    include_charts: bool = Field(True, description="Include performance charts")
    include_validation: bool = Field(True, description="Include validation results")
    custom_sections: Optional[List[str]] = Field(None, description="Custom report sections to include")
    
    class Config:
        schema_extra = {
            "example": {
                "optimization_results": [],
                "report_type": "full",
                "include_charts": True,
                "include_validation": True
            }
        }


class ReportResponse(BaseResponse):
    """Report generation response."""
    report_id: str = Field(..., description="Unique report identifier")
    download_url: str = Field(..., description="URL to download the PDF report")
    pages_count: int = Field(..., description="Number of pages in the report")
    file_size_mb: float = Field(..., description="Report file size in MB")


# Data Management Models
class DataRequest(BaseModel):
    """Data fetching request."""
    asset: Asset = Asset.BTC
    timeframe: TimeFrame = TimeFrame.H4
    start_date: datetime
    end_date: datetime
    exchange: str = Field("binance", description="Exchange to fetch data from")
    use_cache: bool = Field(True, description="Use cached data if available")
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v


class DataResponse(BaseResponse):
    """Data response model."""
    asset: str
    timeframe: str
    records_count: int = Field(..., description="Number of data records")
    date_range: Dict[str, datetime] = Field(..., description="Actual data date range")
    from_cache: bool = Field(..., description="Whether data was served from cache")
    download_url: Optional[str] = Field(None, description="URL to download raw data")


# Health and System Models
class SystemHealth(BaseModel):
    """System health status."""
    status: str = Field(..., description="Overall system status")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    components: Dict[str, str] = Field(..., description="Component health status")
    active_jobs: int = Field(..., description="Number of active optimization jobs")
    queue_size: int = Field(..., description="Job queue size")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")


class SystemMetrics(BaseModel):
    """System performance metrics."""
    requests_total: int = Field(..., description="Total API requests processed")
    requests_per_minute: float = Field(..., description="Current requests per minute")
    avg_response_time_ms: float = Field(..., description="Average response time in milliseconds")
    error_rate_percent: float = Field(..., description="Error rate percentage")
    optimizations_completed: int = Field(..., description="Total optimizations completed")
    optimizations_failed: int = Field(..., description="Total optimizations failed")
    cache_hit_rate: float = Field(..., ge=0, le=1, description="Data cache hit rate")


# Job Management Models
class JobInfo(BaseModel):
    """Job information model."""
    job_id: str
    job_type: str = Field(..., description="Type of job (optimization, validation, etc.)")
    status: OptimizationStatus
    priority: int = Field(..., ge=1, le=10, description="Job priority (1=highest, 10=lowest)")
    created_at: datetime
    started_at: Optional[datetime]
    estimated_completion: Optional[datetime]
    progress: Optional[float] = Field(None, ge=0, le=100)


class JobListResponse(BaseResponse):
    """Job list response."""
    jobs: List[JobInfo]
    total_count: int
    active_count: int
    queued_count: int


# Export-related models

class PineScriptExportRequest(BaseModel):
    """Request model for Pine Script generation."""
    
    strategy_name: str = Field(
        ...,
        description="Name of the trading strategy",
        example="MovingAverageCrossover"
    )
    optimization_results: Dict[str, Any] = Field(
        ...,
        description="Complete optimization results including best parameters"
    )
    output_format: Literal["strategy", "indicator"] = Field(
        default="strategy",
        description="Output format - strategy for live trading or indicator for analysis"
    )
    include_debugging: bool = Field(
        default=True,
        description="Include debugging information and comments in the generated code"
    )
    include_alerts: bool = Field(
        default=True,
        description="Include TradingView alert conditions in the script"
    )
    include_visualization: bool = Field(
        default=True,
        description="Include visual plots and charts in the script"
    )

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PineScriptExportResponse(BaseModel):
    """Response model for Pine Script generation."""
    
    file_id: str = Field(
        ...,
        description="Unique identifier for the generated file"
    )
    filename: str = Field(
        ...,
        description="Suggested filename for the Pine Script"
    )
    file_size: int = Field(
        ...,
        description="Size of the generated file in bytes"
    )
    download_url: str = Field(
        ...,
        description="URL to download the generated Pine Script"
    )
    expires_at: str = Field(
        ...,
        description="ISO timestamp when the file will be automatically deleted"
    )
    script_preview: str = Field(
        ...,
        description="Preview of the first 500 characters of the generated script"
    )
    generation_time: str = Field(
        ...,
        description="ISO timestamp when the script was generated"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "pine_20241129_143022_a1b2c3d4",
                "filename": "MovingAverageCrossover_strategy.pine",
                "file_size": 2847,
                "download_url": "/api/v1/export/download/pine_20241129_143022_a1b2c3d4",
                "expires_at": "2024-11-30T14:30:22Z",
                "script_preview": "//@version=5\nstrategy('Moving Average Crossover - Optimized Strategy', overlay=false)\n\n// Strategy Parameters\nfast_period = input.int(12, 'Fast MA Period', minval=1, maxval=100)\nSlow_period = input.int(26, 'Slow MA Period', minval=1, maxval=200)\n...",
                "generation_time": "2024-11-29T14:30:22Z"
            }
        }


class ReportExportRequest(BaseModel):
    """Request model for PDF report generation."""
    
    strategy_name: str = Field(
        ...,
        description="Name of the trading strategy",
        example="MovingAverageCrossover"
    )
    optimization_results: Dict[str, Any] = Field(
        ...,
        description="Complete optimization results for report generation"
    )
    report_type: Literal["full", "executive", "technical"] = Field(
        default="full",
        description="Type of report to generate - full, executive summary, or technical details"
    )
    include_charts: bool = Field(
        default=True,
        description="Include performance charts and visualizations"
    )
    include_detailed_tables: bool = Field(
        default=True,
        description="Include detailed parameter and metrics tables"
    )

    class Config:
        json_schema_extra = {
            "example": {
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
                "report_type": "full",
                "include_charts": True,
                "include_detailed_tables": True
            }
        }


class ReportExportResponse(BaseModel):
    """Response model for PDF report generation."""
    
    file_id: str = Field(
        ...,
        description="Unique identifier for the generated report"
    )
    filename: str = Field(
        ...,
        description="Suggested filename for the PDF report"
    )
    file_size: int = Field(
        ...,
        description="Size of the generated PDF in bytes"
    )
    download_url: str = Field(
        ...,
        description="URL to download the generated PDF report"
    )
    expires_at: str = Field(
        ...,
        description="ISO timestamp when the file will be automatically deleted"
    )
    report_type: str = Field(
        ...,
        description="Type of report that was generated"
    )
    pages_generated: int = Field(
        ...,
        description="Estimated number of pages in the generated report"
    )
    generation_time: str = Field(
        ...,
        description="ISO timestamp when the report was generated"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "report_20241129_143122_e5f6g7h8",
                "filename": "MovingAverageCrossover_report_full.pdf",
                "file_size": 1247568,
                "download_url": "/api/v1/export/download/report_20241129_143122_e5f6g7h8",
                "expires_at": "2024-11-30T14:31:22Z",
                "report_type": "full",
                "pages_generated": 24,
                "generation_time": "2024-11-29T14:31:22Z"
            }
        }


class FileDownloadResponse(BaseModel):
    """Response model for file listing."""
    
    file_id: str = Field(
        ...,
        description="Unique identifier for the file"
    )
    filename: str = Field(
        ...,
        description="Original filename"
    )
    file_type: str = Field(
        ...,
        description="Type of file (pine, report, etc.)"
    )
    file_size: int = Field(
        ...,
        description="Size of the file in bytes"
    )
    created_at: str = Field(
        ...,
        description="ISO timestamp when the file was created"
    )
    download_url: str = Field(
        ...,
        description="URL to download the file"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "pine_20241129_143022_a1b2c3d4",
                "filename": "MovingAverageCrossover_strategy.pine",
                "file_type": "pine",
                "file_size": 2847,
                "created_at": "2024-11-29T14:30:22Z",
                "download_url": "/api/v1/export/download/pine_20241129_143022_a1b2c3d4"
            }
        }


# Business Models for Executive Features

class BusinessMetrics(BaseModel):
    """Business metrics for executive reporting."""
    total_roi: float = Field(..., description="Total return on investment")
    sharpe_ratio: float = Field(..., description="Risk-adjusted return metric")
    max_drawdown: float = Field(..., description="Maximum portfolio drawdown")
    win_rate: float = Field(..., ge=0, le=1, description="Percentage of profitable trades")
    total_trades: int = Field(..., ge=0, description="Total number of trades executed")
    portfolio_value: float = Field(..., description="Current portfolio value")
    active_strategies: int = Field(..., ge=0, description="Number of active strategies")
    
    class Config:
        schema_extra = {
            "example": {
                "total_roi": 0.2350,
                "sharpe_ratio": 1.85,
                "max_drawdown": 0.0850,
                "win_rate": 0.68,
                "total_trades": 124,
                "portfolio_value": 125000.50,
                "active_strategies": 8
            }
        }


class ROICalculation(BaseModel):
    """ROI calculation result model."""
    strategy_id: str = Field(..., description="Strategy identifier")
    strategy_name: str = Field(..., description="Strategy display name")
    roi_value: float = Field(..., description="Calculated ROI value")
    calculation_method: str = Field(..., description="ROI calculation methodology")
    time_period_days: int = Field(..., ge=1, description="Analysis time period in days")
    benchmark_outperformance: float = Field(..., description="Outperformance vs benchmark")
    confidence_score: float = Field(..., ge=0, le=1, description="Result confidence level")
    risk_adjusted_roi: float = Field(..., description="Risk-adjusted ROI value")
    metadata: Dict[str, Any] = Field(..., description="Additional metrics and data")
    
    class Config:
        schema_extra = {
            "example": {
                "strategy_id": "mac_001",
                "strategy_name": "Moving Average Crossover",
                "roi_value": 0.1850,
                "calculation_method": "compound",
                "time_period_days": 90,
                "benchmark_outperformance": 0.0650,
                "confidence_score": 0.82,
                "risk_adjusted_roi": 0.1620,
                "metadata": {
                    "total_trades": 45,
                    "win_rate": 0.71,
                    "max_drawdown": 0.05,
                    "volatility": 0.12,
                    "sharpe_ratio": 2.1
                }
            }
        }


class ExecutiveSummary(BaseModel):
    """Executive summary for business reporting."""
    period: Dict[str, str] = Field(..., description="Analysis period information")
    summary: BusinessMetrics = Field(..., description="Key performance metrics")
    strategy_performance: Dict[str, Any] = Field(..., description="Strategy-level performance data")
    market_comparison: Dict[str, Any] = Field(..., description="Market benchmark comparison")
    risk_metrics: Dict[str, Any] = Field(..., description="Comprehensive risk analysis")
    recent_activity: List[Dict[str, Any]] = Field(..., description="Recent platform activity")
    insights: List[str] = Field([], description="Key business insights and recommendations")
    
    class Config:
        schema_extra = {
            "example": {
                "period": {
                    "start_date": "2024-01-01T00:00:00Z",
                    "end_date": "2024-03-31T23:59:59Z",
                    "duration_days": 90
                },
                "summary": {
                    "total_roi": 0.2350,
                    "sharpe_ratio": 1.85,
                    "max_drawdown": 0.0850,
                    "win_rate": 0.68,
                    "total_trades": 124,
                    "portfolio_value": 125000.50,
                    "active_strategies": 8
                }
            }
        }


class StrategyComparison(BaseModel):
    """Strategy comparison analysis model."""
    strategies: Dict[str, Dict[str, Any]] = Field(..., description="Strategy comparison data")
    rankings: Dict[str, str] = Field(..., description="Strategy performance rankings")
    summary: Dict[str, str] = Field(..., description="Comparison summary insights")
    metrics_analyzed: List[str] = Field(..., description="Metrics included in comparison")
    date_range: str = Field(..., description="Analysis date range")
    
    class Config:
        schema_extra = {
            "example": {
                "strategies": {
                    "mac_001": {
                        "name": "Moving Average Crossover",
                        "metrics": {
                            "roi": 0.1850,
                            "sharpe": 2.1,
                            "max_drawdown": 0.05
                        }
                    }
                },
                "rankings": {
                    "best_overall": "mac_001",
                    "lowest_risk": "rsi_002"
                },
                "summary": {
                    "best_performing": "mac_001",
                    "most_consistent": "ema_003"
                }
            }
        }


class BusinessDashboardResponse(BaseResponse):
    """Business dashboard response model."""
    dashboard_data: ExecutiveSummary = Field(..., description="Complete dashboard data")
    refresh_interval: int = Field(30, description="Recommended refresh interval in seconds")
    user_role: str = Field(..., description="User role for permission context")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "dashboard_data": {
                    "period": {
                        "start_date": "2024-01-01T00:00:00Z",
                        "end_date": "2024-03-31T23:59:59Z",
                        "duration_days": 90
                    }
                },
                "refresh_interval": 30,
                "user_role": "premium"
            }
        }


class ROICalculationRequest(BaseModel):
    """ROI calculation request model."""
    strategy_ids: List[str] = Field(..., description="List of strategy IDs to analyze")
    calculation_method: Literal["compound", "simple", "annualized", "sharpe_adjusted"] = Field(
        "compound", description="ROI calculation methodology"
    )
    time_period_days: Optional[int] = Field(None, ge=1, le=1095, description="Analysis period in days")
    benchmark: Literal["spy", "nasdaq", "custom"] = Field("spy", description="Benchmark for comparison")
    include_fees: bool = Field(True, description="Include trading fees in calculations")
    risk_free_rate: float = Field(0.03, ge=0, le=0.1, description="Risk-free rate for Sharpe calculations")
    
    class Config:
        schema_extra = {
            "example": {
                "strategy_ids": ["mac_001", "rsi_002", "ema_003"],
                "calculation_method": "compound",
                "time_period_days": 90,
                "benchmark": "spy",
                "include_fees": True,
                "risk_free_rate": 0.03
            }
        }


class ROICalculationResponse(BaseResponse):
    """ROI calculation response model."""
    individual_strategies: List[ROICalculation] = Field(..., description="Individual strategy ROI results")
    portfolio_metrics: Dict[str, Any] = Field(..., description="Portfolio-level metrics")
    insights: Dict[str, Any] = Field(..., description="Business insights and recommendations")
    calculation_parameters: Dict[str, Any] = Field(..., description="Calculation parameters used")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "individual_strategies": [
                    {
                        "strategy_id": "mac_001",
                        "strategy_name": "Moving Average Crossover",
                        "roi_value": 0.1850,
                        "calculation_method": "compound",
                        "time_period_days": 90,
                        "benchmark_outperformance": 0.0650,
                        "confidence_score": 0.82,
                        "risk_adjusted_roi": 0.1620
                    }
                ],
                "portfolio_metrics": {
                    "portfolio_roi": 0.1650,
                    "best_performer": "mac_001",
                    "strategy_count": 3
                }
            }
        }


class BusinessReportRequest(BaseModel):
    """Business report generation request."""
    report_type: Literal["executive_summary", "strategy_comparison", "performance_analysis"] = Field(
        ..., description="Type of business report to generate"
    )
    format: Literal["pdf", "excel", "csv", "json"] = Field("pdf", description="Report output format")
    date_range: str = Field("30d", description="Analysis date range")
    strategy_ids: Optional[List[str]] = Field(None, description="Specific strategies to include")
    include_charts: bool = Field(True, description="Include charts and visualizations")
    include_benchmarks: bool = Field(True, description="Include benchmark comparisons")
    recipient_email: Optional[str] = Field(None, description="Email address for report delivery")
    
    class Config:
        schema_extra = {
            "example": {
                "report_type": "executive_summary",
                "format": "pdf",
                "date_range": "90d",
                "strategy_ids": ["mac_001", "rsi_002"],
                "include_charts": True,
                "include_benchmarks": True,
                "recipient_email": "executive@company.com"
            }
        }


class BusinessReportResponse(BaseResponse):
    """Business report generation response."""
    report_id: str = Field(..., description="Unique report identifier")
    status: Literal["queued", "generating", "completed", "failed"] = Field(..., description="Report generation status")
    estimated_completion: str = Field(..., description="Estimated completion time")
    download_url: Optional[str] = Field(None, description="Download URL when ready")
    format: str = Field(..., description="Report format")
    expires_at: Optional[str] = Field(None, description="Report expiration time")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "report_id": "rpt_exec_20241204_001",
                "status": "queued",
                "estimated_completion": "2024-12-04T18:30:00Z",
                "download_url": None,
                "format": "pdf",
                "expires_at": "2024-12-11T18:25:00Z"
            }
        }


class RealtimeMetricsResponse(BaseResponse):
    """Real-time business metrics response."""
    portfolio: Dict[str, Any] = Field(..., description="Portfolio metrics")
    optimizations: Dict[str, Any] = Field(..., description="Optimization status")
    usage: Dict[str, Any] = Field(..., description="Platform usage metrics")
    system: Dict[str, Any] = Field(..., description="System health metrics")
    alerts: List[Dict[str, Any]] = Field([], description="Active alerts and notifications")
    last_updated: str = Field(..., description="Last update timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "portfolio": {
                    "current_value": 125000.50,
                    "today_pnl": 1250.75,
                    "today_pnl_percent": 1.01
                },
                "optimizations": {
                    "active_count": 3,
                    "queued_count": 1,
                    "running_count": 2
                },
                "system": {
                    "status": "healthy",
                    "api_response_time": "< 100ms"
                },
                "last_updated": "2024-12-04T18:25:30Z"
            }
        }


class BusinessAlertRequest(BaseModel):
    """Business alert setup request."""
    metric: str = Field(..., description="Metric to monitor (roi, drawdown, profit, etc.)")
    threshold: float = Field(..., description="Alert threshold value")
    condition: Literal["above", "below", "equals"] = Field(..., description="Alert condition")
    notification_method: Literal["email", "slack", "webhook"] = Field("email", description="Notification method")
    frequency: Literal["immediate", "daily", "weekly"] = Field("immediate", description="Alert frequency")
    enabled: bool = Field(True, description="Whether alert is active")
    
    class Config:
        schema_extra = {
            "example": {
                "metric": "portfolio_drawdown",
                "threshold": 0.10,
                "condition": "above",
                "notification_method": "email",
                "frequency": "immediate",
                "enabled": True
            }
        }


class BusinessAlertResponse(BaseResponse):
    """Business alert setup response."""
    alert_id: str = Field(..., description="Unique alert identifier")
    metric: str = Field(..., description="Monitored metric")
    threshold: float = Field(..., description="Alert threshold")
    condition: str = Field(..., description="Alert condition")
    notification_method: str = Field(..., description="Notification method")
    created_at: str = Field(..., description="Alert creation timestamp")
    status: Literal["active", "paused", "triggered"] = Field("active", description="Alert status")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "alert_id": "alert_001",
                "metric": "portfolio_drawdown",
                "threshold": 0.10,
                "condition": "above",
                "notification_method": "email",
                "created_at": "2024-12-04T18:25:30Z",
                "status": "active"
            }
        }


class PerformanceMetricsResponse(BaseResponse):
    """Performance metrics response model."""
    data: Dict[str, Any] = Field(..., description="Comprehensive performance metrics data")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2024-12-04T18:25:30Z",
                "data": {
                    "api_performance": {
                        "total_requests": 15420,
                        "average_response_time": 0.125,
                        "cache_hit_rate": 0.78,
                        "error_rate": 0.002,
                        "cached_responses": 12026
                    },
                    "system_performance": {
                        "avg_memory_usage": 65.2,
                        "avg_cpu_usage": 23.8,
                        "max_memory_usage": 78.1,
                        "max_cpu_usage": 45.3
                    },
                    "cache_performance": {
                        "hits": 12026,
                        "misses": 3394,
                        "hit_rate": 0.78,
                        "cache_size": 847
                    },
                    "background_tasks": {
                        "active_count": 3,
                        "tasks": {}
                    },
                    "optimization_recommendations": [
                        "Consider increasing cache TTL for better hit rates"
                    ]
                }
            }
        }


class OptimizationReportResponse(BaseResponse):
    """Optimization report response model."""
    data: Dict[str, Any] = Field(..., description="Comprehensive optimization report data")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2024-12-04T18:25:30Z",
                "data": {
                    "cache_performance": {
                        "hits": 12026,
                        "misses": 3394,
                        "hit_rate": 0.78
                    },
                    "system_performance": {
                        "window_minutes": 60,
                        "avg_memory_usage": 65.2,
                        "avg_cpu_usage": 23.8
                    },
                    "memory_analysis": {
                        "snapshots_count": 15,
                        "latest_snapshot_time": "2024-12-04T18:20:30Z",
                        "memory_differences": []
                    },
                    "optimization_recommendations": [
                        "Consider API response time optimization",
                        "Monitor memory usage patterns"
                    ]
                }
            }
        }


class SystemHealthResponse(BaseResponse):
    """System health response model."""
    data: Dict[str, Any] = Field(..., description="Comprehensive system health data")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2024-12-04T18:25:30Z",
                "data": {
                    "status": "healthy",
                    "system_metrics": {
                        "memory_usage_percent": 65.2,
                        "cpu_usage_percent": 23.8,
                        "memory_available_gb": 8.5,
                        "disk_read_mb_per_sec": 12.3,
                        "disk_write_mb_per_sec": 8.7
                    },
                    "cache_stats": {
                        "hits": 12026,
                        "misses": 3394,
                        "hit_rate": 0.78
                    },
                    "health_issues": [],
                    "uptime_info": {
                        "memory_available_gb": 8.5,
                        "cpu_usage_percent": 23.8,
                        "memory_usage_percent": 65.2
                    }
                }
            }
        }


class StandardResponse(BaseResponse):
    """Standard response model for simple operations."""
    pass 
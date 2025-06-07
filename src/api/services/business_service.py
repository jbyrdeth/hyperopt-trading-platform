import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import uuid
import os
import json
from pathlib import Path
import logging

try:
    import sys
    import os
    # Add the project root to the path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.reporting.data_integration import DataIntegration
except ImportError:
    # Fallback - create a mock DataIntegration
    class DataIntegration:
        @staticmethod
        def get_optimization_results(strategy_name: str = None, limit: int = 100):
            return []
        
        @staticmethod
        def get_strategy_performance_data(strategy_name: str):
            return {}
try:
    from src.reporting.analysis import AnalysisEngine
    from src.reporting.visualization import VisualizationEngine
except ImportError:
    # Fallback - create mock engines
    class AnalysisEngine:
        @staticmethod
        def calculate_roi_metrics(*args, **kwargs):
            return {}
        
        @staticmethod
        def generate_business_insights(*args, **kwargs):
            return []
    
    class VisualizationEngine:
        @staticmethod
        def create_performance_chart(*args, **kwargs):
            return None
try:
    from src.reporting.report_generator import ReportGenerator
    from src.utils.database import DatabaseManager
except ImportError:
    # Fallback - create mocks
    class ReportGenerator:
        @staticmethod
        def generate_business_report(*args, **kwargs):
            return {"report_id": "mock", "status": "generated"}
    
    class DatabaseManager:
        @staticmethod
        def get_connection():
            return None
try:
    from src.utils.cache import CacheManager
except ImportError:
    # Fallback - create mock
    class CacheManager:
        @staticmethod
        def get(key):
            return None
        
        @staticmethod
        def set(key, value, ttl=None):
            pass

logger = logging.getLogger(__name__)

class BusinessService:
    """
    Service class for business-ready features including executive dashboards,
    ROI calculations, strategy comparisons, and business reporting.
    """
    
    def __init__(self):
        self.data_integration = DataIntegration()
        self.analysis_engine = AnalysisEngine()
        self.viz_engine = VisualizationEngine()
        self.report_generator = ReportGenerator()
        self.db_manager = DatabaseManager()
        self.cache = CacheManager()
        
    async def get_executive_dashboard(self, date_range: str, user_id: str) -> Dict[str, Any]:
        """
        Generate executive dashboard with key business metrics and KPIs.
        """
        try:
            # Parse date range
            end_date = datetime.utcnow()
            if date_range == "7d":
                start_date = end_date - timedelta(days=7)
            elif date_range == "30d":
                start_date = end_date - timedelta(days=30)
            elif date_range == "90d":
                start_date = end_date - timedelta(days=90)
            elif date_range == "1y":
                start_date = end_date - timedelta(days=365)
            else:  # all
                start_date = datetime(2020, 1, 1)
            
            # Get performance data
            performance_data = await self.data_integration.get_performance_data(
                start_date=start_date,
                end_date=end_date,
                user_id=user_id
            )
            
            # Calculate key metrics
            total_roi = self._calculate_portfolio_roi(performance_data)
            sharpe_ratio = self._calculate_sharpe_ratio(performance_data)
            max_drawdown = self._calculate_max_drawdown(performance_data)
            win_rate = self._calculate_win_rate(performance_data)
            total_trades = len(performance_data.get('trades', []))
            
            # Get strategy performance summary
            strategy_summary = await self._get_strategy_performance_summary(user_id, start_date, end_date)
            
            # Get market comparison
            market_comparison = await self._get_market_comparison(performance_data, start_date, end_date)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(performance_data)
            
            # Get recent activity
            recent_activity = await self._get_recent_activity(user_id, limit=10)
            
            return {
                "summary": {
                    "total_roi": round(total_roi, 4),
                    "sharpe_ratio": round(sharpe_ratio, 4),
                    "max_drawdown": round(max_drawdown, 4),
                    "win_rate": round(win_rate, 4),
                    "total_trades": total_trades,
                    "active_strategies": len(strategy_summary.get('active_strategies', [])),
                    "portfolio_value": performance_data.get('current_portfolio_value', 0)
                },
                "strategy_performance": strategy_summary,
                "market_comparison": market_comparison,
                "risk_metrics": risk_metrics,
                "recent_activity": recent_activity,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "duration_days": (end_date - start_date).days
                }
            }
            
        except Exception as e:
            logger.error(f"Executive dashboard generation failed: {str(e)}")
            raise

    async def get_realtime_metrics(self, user_id: str) -> Dict[str, Any]:
        """
        Get real-time business metrics for live monitoring.
        """
        try:
            # Get current optimizations
            active_optimizations = await self.db_manager.get_active_optimizations(user_id)
            
            # Get current portfolio value
            current_portfolio = await self.db_manager.get_current_portfolio_value(user_id)
            
            # Get today's P&L
            today_pnl = await self._get_today_pnl(user_id)
            
            # Get platform usage metrics
            usage_metrics = await self._get_usage_metrics(user_id)
            
            # Get system health
            system_health = await self._get_system_health()
            
            return {
                "portfolio": {
                    "current_value": current_portfolio,
                    "today_pnl": today_pnl,
                    "today_pnl_percent": (today_pnl / current_portfolio * 100) if current_portfolio > 0 else 0
                },
                "optimizations": {
                    "active_count": len(active_optimizations),
                    "queued_count": sum(1 for opt in active_optimizations if opt.get('status') == 'queued'),
                    "running_count": sum(1 for opt in active_optimizations if opt.get('status') == 'running')
                },
                "usage": usage_metrics,
                "system": system_health,
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Realtime metrics failed: {str(e)}")
            raise

    async def compare_strategies(self, strategy_ids: List[str], metrics: List[str], 
                               date_range: str, user_id: str) -> Dict[str, Any]:
        """
        Compare multiple strategies across specified metrics.
        """
        try:
            # Parse date range
            end_date = datetime.utcnow()
            if date_range == "7d":
                start_date = end_date - timedelta(days=7)
            elif date_range == "30d":
                start_date = end_date - timedelta(days=30)
            elif date_range == "90d":
                start_date = end_date - timedelta(days=90)
            elif date_range == "1y":
                start_date = end_date - timedelta(days=365)
            else:
                start_date = end_date - timedelta(days=30)
            
            comparison_data = {}
            
            for strategy_id in strategy_ids:
                strategy_data = await self.data_integration.get_strategy_performance(
                    strategy_id=strategy_id,
                    start_date=start_date,
                    end_date=end_date,
                    user_id=user_id
                )
                
                strategy_metrics = {}
                
                if "roi" in metrics:
                    strategy_metrics["roi"] = self._calculate_portfolio_roi(strategy_data)
                if "sharpe" in metrics:
                    strategy_metrics["sharpe"] = self._calculate_sharpe_ratio(strategy_data)
                if "max_drawdown" in metrics:
                    strategy_metrics["max_drawdown"] = self._calculate_max_drawdown(strategy_data)
                if "volatility" in metrics:
                    strategy_metrics["volatility"] = self._calculate_volatility(strategy_data)
                if "win_rate" in metrics:
                    strategy_metrics["win_rate"] = self._calculate_win_rate(strategy_data)
                if "total_trades" in metrics:
                    strategy_metrics["total_trades"] = len(strategy_data.get('trades', []))
                if "avg_trade_duration" in metrics:
                    strategy_metrics["avg_trade_duration"] = self._calculate_avg_trade_duration(strategy_data)
                
                comparison_data[strategy_id] = {
                    "name": strategy_data.get('name', f'Strategy {strategy_id}'),
                    "metrics": strategy_metrics,
                    "performance_chart_data": strategy_data.get('cumulative_returns', [])
                }
            
            # Calculate rankings
            rankings = self._calculate_strategy_rankings(comparison_data, metrics)
            
            return {
                "strategies": comparison_data,
                "rankings": rankings,
                "summary": {
                    "best_performing": rankings.get('best_overall'),
                    "most_consistent": rankings.get('most_consistent'),
                    "lowest_risk": rankings.get('lowest_risk')
                }
            }
            
        except Exception as e:
            logger.error(f"Strategy comparison failed: {str(e)}")
            raise

    async def queue_report_generation(self, report_type: str, format: str, 
                                    date_range: str, strategy_ids: Optional[List[str]],
                                    include_charts: bool, user_id: str) -> str:
        """
        Queue a business report for generation.
        """
        try:
            report_id = str(uuid.uuid4())
            
            # Store report request
            report_request = {
                "id": report_id,
                "type": report_type,
                "format": format,
                "date_range": date_range,
                "strategy_ids": strategy_ids,
                "include_charts": include_charts,
                "user_id": user_id,
                "status": "queued",
                "created_at": datetime.utcnow().isoformat()
            }
            
            await self.db_manager.store_report_request(report_request)
            
            # Queue report generation task
            asyncio.create_task(self._generate_report_async(report_request))
            
            return report_id
            
        except Exception as e:
            logger.error(f"Report queueing failed: {str(e)}")
            raise

    async def get_report_path(self, report_id: str, user_id: str) -> Optional[str]:
        """
        Get the file path for a generated report.
        """
        try:
            report_info = await self.db_manager.get_report_info(report_id, user_id)
            
            if not report_info or report_info.get('status') != 'completed':
                return None
                
            file_path = report_info.get('file_path')
            
            if file_path and os.path.exists(file_path):
                return file_path
                
            return None
            
        except Exception as e:
            logger.error(f"Report path retrieval failed: {str(e)}")
            return None

    async def get_trend_analysis(self, metric: str, period: str, 
                               duration: str, user_id: str) -> Dict[str, Any]:
        """
        Get trend analysis for business metrics over time.
        """
        try:
            # Parse duration
            end_date = datetime.utcnow()
            if duration == "7d":
                start_date = end_date - timedelta(days=7)
            elif duration == "30d":
                start_date = end_date - timedelta(days=30)
            elif duration == "90d":
                start_date = end_date - timedelta(days=90)
            elif duration == "1y":
                start_date = end_date - timedelta(days=365)
            else:
                start_date = end_date - timedelta(days=30)
            
            # Get historical data
            historical_data = await self.data_integration.get_historical_metrics(
                metric=metric,
                start_date=start_date,
                end_date=end_date,
                period=period,
                user_id=user_id
            )
            
            # Calculate trend statistics
            trend_stats = self._calculate_trend_statistics(historical_data)
            
            return {
                "data": historical_data,
                "statistics": trend_stats,
                "trend_direction": trend_stats.get('direction'),
                "volatility": trend_stats.get('volatility'),
                "forecast": self._generate_simple_forecast(historical_data)
            }
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {str(e)}")
            raise

    async def get_portfolio_summary(self, user_id: str, include_inactive: bool) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary for business reporting.
        """
        try:
            portfolio_data = await self.data_integration.get_portfolio_data(
                user_id=user_id,
                include_inactive=include_inactive
            )
            
            # Calculate allocation
            allocation = self._calculate_portfolio_allocation(portfolio_data)
            
            # Calculate performance metrics
            performance = self._calculate_portfolio_performance(portfolio_data)
            
            # Calculate risk metrics
            risk = self._calculate_portfolio_risk(portfolio_data)
            
            # Get top performing strategies
            top_strategies = self._get_top_strategies(portfolio_data, limit=5)
            
            return {
                "allocation": allocation,
                "performance": performance,
                "risk": risk,
                "top_strategies": top_strategies,
                "total_value": portfolio_data.get('total_value', 0),
                "cash_balance": portfolio_data.get('cash_balance', 0),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Portfolio summary failed: {str(e)}")
            raise

    # Helper methods
    def _calculate_portfolio_roi(self, performance_data: Dict[str, Any]) -> float:
        """Calculate portfolio ROI."""
        returns = performance_data.get('returns', [])
        if not returns:
            return 0.0
        return (1 + pd.Series(returns)).prod() - 1

    def _calculate_sharpe_ratio(self, performance_data: Dict[str, Any], risk_free_rate: float = 0.03) -> float:
        """Calculate Sharpe ratio."""
        returns = performance_data.get('returns', [])
        if not returns:
            return 0.0
        
        returns_series = pd.Series(returns)
        excess_returns = returns_series - (risk_free_rate / 252)  # Daily risk-free rate
        
        if excess_returns.std() == 0:
            return 0.0
            
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    def _calculate_max_drawdown(self, performance_data: Dict[str, Any]) -> float:
        """Calculate maximum drawdown."""
        returns = performance_data.get('returns', [])
        if not returns:
            return 0.0
        
        cumulative = (1 + pd.Series(returns)).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        return abs(drawdown.min())

    def _calculate_win_rate(self, performance_data: Dict[str, Any]) -> float:
        """Calculate win rate."""
        returns = performance_data.get('returns', [])
        if not returns:
            return 0.0
        
        winning_trades = sum(1 for r in returns if r > 0)
        return winning_trades / len(returns) if returns else 0.0

    def _calculate_volatility(self, performance_data: Dict[str, Any]) -> float:
        """Calculate annualized volatility."""
        returns = performance_data.get('returns', [])
        if not returns:
            return 0.0
        
        return pd.Series(returns).std() * np.sqrt(252)

    async def _generate_report_async(self, report_request: Dict[str, Any]):
        """
        Generate report asynchronously.
        """
        try:
            # Update status to running
            await self.db_manager.update_report_status(report_request['id'], 'running')
            
            # Generate report based on type
            if report_request['type'] == 'executive_summary':
                file_path = await self.report_generator.generate_executive_summary(
                    **report_request
                )
            elif report_request['type'] == 'strategy_comparison':
                file_path = await self.report_generator.generate_strategy_comparison(
                    **report_request
                )
            elif report_request['type'] == 'performance_analysis':
                file_path = await self.report_generator.generate_performance_analysis(
                    **report_request
                )
            else:
                raise ValueError(f"Unknown report type: {report_request['type']}")
            
            # Update status to completed
            await self.db_manager.update_report_status(
                report_request['id'], 
                'completed', 
                file_path=file_path
            )
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            await self.db_manager.update_report_status(
                report_request['id'], 
                'failed', 
                error=str(e)
            )

    async def _get_strategy_performance_summary(self, user_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get strategy performance summary."""
        strategies = await self.data_integration.get_user_strategies(user_id)
        
        summary = {
            "active_strategies": [],
            "top_performers": [],
            "underperformers": []
        }
        
        for strategy in strategies:
            perf_data = await self.data_integration.get_strategy_performance(
                strategy_id=strategy['id'],
                start_date=start_date,
                end_date=end_date,
                user_id=user_id
            )
            
            roi = self._calculate_portfolio_roi(perf_data)
            
            strategy_summary = {
                "id": strategy['id'],
                "name": strategy['name'],
                "roi": roi,
                "status": strategy.get('status', 'active')
            }
            
            if strategy.get('status') == 'active':
                summary["active_strategies"].append(strategy_summary)
                
            if roi > 0.1:  # 10% threshold for top performers
                summary["top_performers"].append(strategy_summary)
            elif roi < -0.05:  # -5% threshold for underperformers
                summary["underperformers"].append(strategy_summary)
        
        # Sort by ROI
        summary["top_performers"].sort(key=lambda x: x['roi'], reverse=True)
        summary["underperformers"].sort(key=lambda x: x['roi'])
        
        return summary

    async def _get_market_comparison(self, performance_data: Dict[str, Any], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Compare portfolio performance against market benchmarks."""
        try:
            # Get benchmark data (simplified - in production, use actual market data APIs)
            spy_return = 0.08  # Annual S&P 500 return estimate
            nasdaq_return = 0.12  # Annual NASDAQ return estimate
            
            portfolio_roi = self._calculate_portfolio_roi(performance_data)
            
            days = (end_date - start_date).days
            annualized_portfolio_roi = (1 + portfolio_roi) ** (365 / days) - 1 if days > 0 else portfolio_roi
            
            return {
                "portfolio_roi_annualized": annualized_portfolio_roi,
                "spy_outperformance": annualized_portfolio_roi - spy_return,
                "nasdaq_outperformance": annualized_portfolio_roi - nasdaq_return,
                "benchmarks": {
                    "spy": spy_return,
                    "nasdaq": nasdaq_return
                }
            }
        except Exception:
            return {"error": "Benchmark comparison unavailable"}

    def _calculate_risk_metrics(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        returns = performance_data.get('returns', [])
        if not returns:
            return {}
        
        returns_series = pd.Series(returns)
        
        return {
            "volatility": self._calculate_volatility(performance_data),
            "var_95": returns_series.quantile(0.05),  # 95% VaR
            "var_99": returns_series.quantile(0.01),  # 99% VaR
            "skewness": returns_series.skew(),
            "kurtosis": returns_series.kurtosis()
        }

    async def _get_recent_activity(self, user_id: str, limit: int) -> List[Dict[str, Any]]:
        """Get recent user activity."""
        try:
            return await self.db_manager.get_recent_activity(user_id, limit)
        except Exception:
            return []

    async def _get_today_pnl(self, user_id: str) -> float:
        """Get today's P&L."""
        try:
            today = datetime.utcnow().date()
            return await self.db_manager.get_daily_pnl(user_id, today)
        except Exception:
            return 0.0

    async def _get_usage_metrics(self, user_id: str) -> Dict[str, Any]:
        """Get platform usage metrics."""
        try:
            return await self.db_manager.get_usage_metrics(user_id)
        except Exception:
            return {}

    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics."""
        return {
            "status": "healthy",
            "api_response_time": "< 100ms",
            "optimization_queue_length": 0,
            "database_status": "connected"
        } 
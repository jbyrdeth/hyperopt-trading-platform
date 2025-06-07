import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

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
    from src.utils.database import DatabaseManager
except ImportError:
    # Fallback - create mock
    class DatabaseManager:
        @staticmethod
        def get_connection():
            return None

logger = logging.getLogger(__name__)

@dataclass
class ROIResult:
    """Data class for ROI calculation results."""
    strategy_id: str
    strategy_name: str
    roi_value: float
    calculation_method: str
    time_period_days: int
    benchmark_outperformance: float
    confidence_score: float
    risk_adjusted_roi: float
    metadata: Dict[str, Any]

class ROICalculator:
    """
    Comprehensive ROI calculator supporting multiple methodologies
    and business-focused metrics for decision making.
    """
    
    def __init__(self):
        self.data_integration = DataIntegration()
        self.db_manager = DatabaseManager()
        
    async def calculate_comprehensive_roi(self, strategy_ids: List[str], 
                                        calculation_method: str,
                                        user_id: str,
                                        time_period_days: Optional[int] = None,
                                        benchmark: str = "spy",
                                        include_fees: bool = True,
                                        risk_free_rate: float = 0.03) -> Dict[str, Any]:
        """
        Calculate comprehensive ROI for multiple strategies using specified methodology.
        """
        try:
            results = []
            
            for strategy_id in strategy_ids:
                roi_result = await self._calculate_strategy_roi(
                    strategy_id=strategy_id,
                    calculation_method=calculation_method,
                    time_period_days=time_period_days,
                    benchmark=benchmark,
                    include_fees=include_fees,
                    risk_free_rate=risk_free_rate,
                    user_id=user_id
                )
                results.append(roi_result)
            
            # Calculate portfolio-level metrics
            portfolio_metrics = self._calculate_portfolio_level_metrics(results)
            
            # Generate insights and recommendations
            insights = self._generate_roi_insights(results, calculation_method)
            
            return {
                "individual_strategies": [result.__dict__ for result in results],
                "portfolio_metrics": portfolio_metrics,
                "insights": insights,
                "calculation_parameters": {
                    "method": calculation_method,
                    "time_period_days": time_period_days,
                    "benchmark": benchmark,
                    "includes_fees": include_fees,
                    "risk_free_rate": risk_free_rate
                }
            }
            
        except Exception as e:
            logger.error(f"Comprehensive ROI calculation failed: {str(e)}")
            raise

    async def _calculate_strategy_roi(self, strategy_id: str, calculation_method: str,
                                    time_period_days: Optional[int], benchmark: str,
                                    include_fees: bool, risk_free_rate: float,
                                    user_id: str) -> ROIResult:
        """
        Calculate ROI for a single strategy using specified methodology.
        """
        try:
            # Get strategy performance data
            end_date = datetime.utcnow()
            
            if time_period_days:
                start_date = end_date - timedelta(days=time_period_days)
            else:
                # Use strategy inception date or default to 1 year
                start_date = end_date - timedelta(days=365)
            
            strategy_data = await self.data_integration.get_strategy_performance(
                strategy_id=strategy_id,
                start_date=start_date,
                end_date=end_date,
                user_id=user_id
            )
            
            # Calculate ROI based on method
            if calculation_method == "compound":
                roi_value = self._calculate_compound_roi(strategy_data, include_fees)
            elif calculation_method == "simple":
                roi_value = self._calculate_simple_roi(strategy_data, include_fees)
            elif calculation_method == "annualized":
                roi_value = self._calculate_annualized_roi(strategy_data, include_fees, start_date, end_date)
            elif calculation_method == "sharpe_adjusted":
                roi_value = self._calculate_sharpe_adjusted_roi(strategy_data, include_fees, risk_free_rate)
            else:
                raise ValueError(f"Unknown calculation method: {calculation_method}")
            
            # Calculate benchmark outperformance
            benchmark_roi = await self._get_benchmark_roi(benchmark, start_date, end_date)
            benchmark_outperformance = roi_value - benchmark_roi
            
            # Calculate risk-adjusted ROI
            risk_adjusted_roi = self._calculate_risk_adjusted_roi(strategy_data, roi_value, risk_free_rate)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(strategy_data)
            
            # Prepare metadata
            metadata = {
                "total_trades": len(strategy_data.get('trades', [])),
                "win_rate": self._calculate_win_rate(strategy_data),
                "max_drawdown": self._calculate_max_drawdown(strategy_data),
                "volatility": self._calculate_volatility(strategy_data),
                "sharpe_ratio": self._calculate_sharpe_ratio(strategy_data, risk_free_rate),
                "sortino_ratio": self._calculate_sortino_ratio(strategy_data, risk_free_rate),
                "calmar_ratio": self._calculate_calmar_ratio(strategy_data),
                "beta": await self._calculate_beta(strategy_data, benchmark, start_date, end_date),
                "alpha": self._calculate_alpha(roi_value, benchmark_roi, strategy_data, benchmark),
                "information_ratio": self._calculate_information_ratio(strategy_data, benchmark_roi)
            }
            
            return ROIResult(
                strategy_id=strategy_id,
                strategy_name=strategy_data.get('name', f'Strategy {strategy_id}'),
                roi_value=roi_value,
                calculation_method=calculation_method,
                time_period_days=(end_date - start_date).days,
                benchmark_outperformance=benchmark_outperformance,
                confidence_score=confidence_score,
                risk_adjusted_roi=risk_adjusted_roi,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Strategy ROI calculation failed for {strategy_id}: {str(e)}")
            raise

    def _calculate_compound_roi(self, strategy_data: Dict[str, Any], include_fees: bool) -> float:
        """Calculate compound annual growth rate (CAGR)."""
        returns = strategy_data.get('returns', [])
        if not returns:
            return 0.0
        
        # Apply fees if requested
        if include_fees:
            fee_rate = strategy_data.get('fee_rate', 0.001)  # 0.1% default fee
            returns = [r - fee_rate for r in returns]
        
        # Calculate compound return
        cumulative_return = (1 + pd.Series(returns)).prod()
        return cumulative_return - 1

    def _calculate_simple_roi(self, strategy_data: Dict[str, Any], include_fees: bool) -> float:
        """Calculate simple ROI."""
        returns = strategy_data.get('returns', [])
        if not returns:
            return 0.0
        
        # Apply fees if requested
        if include_fees:
            fee_rate = strategy_data.get('fee_rate', 0.001)
            returns = [r - fee_rate for r in returns]
        
        return sum(returns)

    def _calculate_annualized_roi(self, strategy_data: Dict[str, Any], include_fees: bool,
                                start_date: datetime, end_date: datetime) -> float:
        """Calculate annualized ROI."""
        compound_roi = self._calculate_compound_roi(strategy_data, include_fees)
        
        days = (end_date - start_date).days
        if days <= 0:
            return compound_roi
        
        years = days / 365.25
        if years <= 0:
            return compound_roi
        
        annualized_roi = (1 + compound_roi) ** (1 / years) - 1
        return annualized_roi

    def _calculate_sharpe_adjusted_roi(self, strategy_data: Dict[str, Any], 
                                     include_fees: bool, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio adjusted ROI."""
        compound_roi = self._calculate_compound_roi(strategy_data, include_fees)
        sharpe_ratio = self._calculate_sharpe_ratio(strategy_data, risk_free_rate)
        
        # Adjust ROI by Sharpe ratio (higher Sharpe = better risk-adjusted return)
        sharpe_adjustment = max(0, min(2, sharpe_ratio / 2))  # Cap adjustment factor
        return compound_roi * sharpe_adjustment

    def _calculate_risk_adjusted_roi(self, strategy_data: Dict[str, Any], 
                                   roi_value: float, risk_free_rate: float) -> float:
        """Calculate risk-adjusted ROI using multiple risk metrics."""
        volatility = self._calculate_volatility(strategy_data)
        max_drawdown = self._calculate_max_drawdown(strategy_data)
        
        # Risk adjustment factor based on volatility and drawdown
        volatility_penalty = max(0, volatility - 0.15)  # Penalty for vol > 15%
        drawdown_penalty = max(0, max_drawdown - 0.10)  # Penalty for drawdown > 10%
        
        risk_adjustment = 1 - (volatility_penalty * 0.5 + drawdown_penalty * 0.3)
        risk_adjustment = max(0.1, risk_adjustment)  # Minimum 10% of original ROI
        
        return roi_value * risk_adjustment

    def _calculate_confidence_score(self, strategy_data: Dict[str, Any]) -> float:
        """Calculate confidence score based on data quality and consistency."""
        returns = strategy_data.get('returns', [])
        if not returns:
            return 0.0
        
        # Factors affecting confidence
        trade_count = len(strategy_data.get('trades', []))
        data_points = len(returns)
        win_rate = self._calculate_win_rate(strategy_data)
        consistency = 1 - (pd.Series(returns).std() / abs(pd.Series(returns).mean())) if pd.Series(returns).mean() != 0 else 0
        
        # Scoring components (0-1 scale)
        trade_score = min(1, trade_count / 100)  # Max score at 100+ trades
        data_score = min(1, data_points / 252)   # Max score at 1 year of data
        win_rate_score = win_rate
        consistency_score = max(0, min(1, consistency))
        
        # Weighted average
        confidence = (trade_score * 0.3 + data_score * 0.3 + 
                     win_rate_score * 0.2 + consistency_score * 0.2)
        
        return confidence

    async def _get_benchmark_roi(self, benchmark: str, start_date: datetime, end_date: datetime) -> float:
        """Get benchmark ROI for comparison."""
        try:
            if benchmark == "spy":
                # Simplified S&P 500 annual return
                days = (end_date - start_date).days
                annual_return = 0.10  # 10% annual estimate
                return (annual_return * days / 365.25)
            elif benchmark == "nasdaq":
                # Simplified NASDAQ annual return
                days = (end_date - start_date).days
                annual_return = 0.12  # 12% annual estimate
                return (annual_return * days / 365.25)
            else:
                # Custom benchmark - get from database
                benchmark_data = await self.db_manager.get_benchmark_data(benchmark, start_date, end_date)
                if benchmark_data:
                    return self._calculate_compound_roi(benchmark_data, include_fees=False)
                return 0.0
        except Exception:
            return 0.0

    def _calculate_portfolio_level_metrics(self, results: List[ROIResult]) -> Dict[str, Any]:
        """Calculate portfolio-level ROI metrics."""
        if not results:
            return {}
        
        # Weighted average ROI (equal weights for now)
        avg_roi = sum(result.roi_value for result in results) / len(results)
        
        # Portfolio volatility (simplified)
        roi_values = [result.roi_value for result in results]
        portfolio_volatility = pd.Series(roi_values).std() if len(roi_values) > 1 else 0
        
        # Best and worst performers
        best_performer = max(results, key=lambda x: x.roi_value)
        worst_performer = min(results, key=lambda x: x.roi_value)
        
        # Risk-adjusted metrics
        avg_sharpe = np.mean([result.metadata.get('sharpe_ratio', 0) for result in results])
        avg_max_drawdown = np.mean([result.metadata.get('max_drawdown', 0) for result in results])
        
        return {
            "portfolio_roi": avg_roi,
            "portfolio_volatility": portfolio_volatility,
            "best_performer": {
                "strategy_id": best_performer.strategy_id,
                "strategy_name": best_performer.strategy_name,
                "roi": best_performer.roi_value
            },
            "worst_performer": {
                "strategy_id": worst_performer.strategy_id,
                "strategy_name": worst_performer.strategy_name,
                "roi": worst_performer.roi_value
            },
            "average_sharpe_ratio": avg_sharpe,
            "average_max_drawdown": avg_max_drawdown,
            "strategy_count": len(results),
            "correlation_risk": self._calculate_correlation_risk(results)
        }

    def _generate_roi_insights(self, results: List[ROIResult], calculation_method: str) -> Dict[str, Any]:
        """Generate business insights from ROI calculations."""
        if not results:
            return {}
        
        insights = {
            "summary": f"Analyzed {len(results)} strategies using {calculation_method} methodology",
            "recommendations": [],
            "risk_alerts": [],
            "opportunities": []
        }
        
        # Performance insights
        positive_roi_count = sum(1 for r in results if r.roi_value > 0)
        if positive_roi_count / len(results) > 0.8:
            insights["recommendations"].append("Strong portfolio performance with 80%+ strategies profitable")
        elif positive_roi_count / len(results) < 0.5:
            insights["risk_alerts"].append("Less than 50% of strategies are profitable - review allocation")
        
        # Risk insights
        high_risk_strategies = [r for r in results if r.metadata.get('max_drawdown', 0) > 0.2]
        if high_risk_strategies:
            insights["risk_alerts"].append(f"{len(high_risk_strategies)} strategies have >20% max drawdown")
        
        # Opportunity insights
        low_confidence_high_roi = [r for r in results if r.roi_value > 0.15 and r.confidence_score < 0.5]
        if low_confidence_high_roi:
            insights["opportunities"].append(f"{len(low_confidence_high_roi)} high-ROI strategies need more data for validation")
        
        return insights

    # Helper calculation methods
    def _calculate_win_rate(self, strategy_data: Dict[str, Any]) -> float:
        """Calculate win rate."""
        returns = strategy_data.get('returns', [])
        if not returns:
            return 0.0
        
        winning_trades = sum(1 for r in returns if r > 0)
        return winning_trades / len(returns)

    def _calculate_max_drawdown(self, strategy_data: Dict[str, Any]) -> float:
        """Calculate maximum drawdown."""
        returns = strategy_data.get('returns', [])
        if not returns:
            return 0.0
        
        cumulative = (1 + pd.Series(returns)).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        return abs(drawdown.min())

    def _calculate_volatility(self, strategy_data: Dict[str, Any]) -> float:
        """Calculate annualized volatility."""
        returns = strategy_data.get('returns', [])
        if not returns:
            return 0.0
        
        return pd.Series(returns).std() * np.sqrt(252)

    def _calculate_sharpe_ratio(self, strategy_data: Dict[str, Any], risk_free_rate: float) -> float:
        """Calculate Sharpe ratio."""
        returns = strategy_data.get('returns', [])
        if not returns:
            return 0.0
        
        returns_series = pd.Series(returns)
        excess_returns = returns_series - (risk_free_rate / 252)
        
        if excess_returns.std() == 0:
            return 0.0
            
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    def _calculate_sortino_ratio(self, strategy_data: Dict[str, Any], risk_free_rate: float) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        returns = strategy_data.get('returns', [])
        if not returns:
            return 0.0
        
        returns_series = pd.Series(returns)
        excess_returns = returns_series - (risk_free_rate / 252)
        
        # Only consider negative returns for downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
            
        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)

    def _calculate_calmar_ratio(self, strategy_data: Dict[str, Any]) -> float:
        """Calculate Calmar ratio (return/max drawdown)."""
        returns = strategy_data.get('returns', [])
        if not returns:
            return 0.0
        
        annual_return = self._calculate_annualized_roi(strategy_data, False, 
                                                      datetime.utcnow() - timedelta(days=365), 
                                                      datetime.utcnow())
        max_drawdown = self._calculate_max_drawdown(strategy_data)
        
        if max_drawdown == 0:
            return float('inf') if annual_return > 0 else 0.0
            
        return annual_return / max_drawdown

    async def _calculate_beta(self, strategy_data: Dict[str, Any], benchmark: str,
                            start_date: datetime, end_date: datetime) -> float:
        """Calculate beta relative to benchmark."""
        try:
            strategy_returns = strategy_data.get('returns', [])
            benchmark_roi = await self._get_benchmark_roi(benchmark, start_date, end_date)
            
            # Simplified beta calculation (in production, use proper correlation)
            strategy_volatility = self._calculate_volatility(strategy_data)
            market_volatility = 0.16  # Assumed market volatility
            
            # Simplified beta estimate
            return strategy_volatility / market_volatility if market_volatility > 0 else 1.0
        except Exception:
            return 1.0

    def _calculate_alpha(self, strategy_roi: float, benchmark_roi: float, 
                        strategy_data: Dict[str, Any], benchmark: str) -> float:
        """Calculate alpha (excess return vs benchmark)."""
        # Simplified alpha = strategy return - benchmark return
        return strategy_roi - benchmark_roi

    def _calculate_information_ratio(self, strategy_data: Dict[str, Any], benchmark_roi: float) -> float:
        """Calculate information ratio."""
        strategy_roi = self._calculate_compound_roi(strategy_data, False)
        excess_return = strategy_roi - benchmark_roi
        
        # Simplified tracking error (use strategy volatility as proxy)
        tracking_error = self._calculate_volatility(strategy_data)
        
        if tracking_error == 0:
            return 0.0
            
        return excess_return / tracking_error

    def _calculate_correlation_risk(self, results: List[ROIResult]) -> float:
        """Calculate correlation risk of portfolio strategies."""
        if len(results) < 2:
            return 0.0
        
        # Simplified correlation risk based on ROI similarity
        roi_values = [result.roi_value for result in results]
        return pd.Series(roi_values).std() / abs(pd.Series(roi_values).mean()) if pd.Series(roi_values).mean() != 0 else 0 
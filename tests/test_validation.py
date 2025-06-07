"""Quick validation tests for a few trading strategies."""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure the project root is on the path so the ``src`` package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from hyperopt.pyll import stochastic

from src.optimization.strategy_factory import StrategyFactory
from src.strategies.backtesting_engine import BacktestingEngine


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    """Create synthetic data with clear trends."""
    dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq="1h")
    np.random.seed(42)
    trend = np.linspace(100, 130, len(dates))
    prices = trend + np.sin(np.linspace(0, 10 * np.pi, len(dates))) * 5
    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "close": prices,
            "volume": np.random.uniform(1000, 10000, len(dates)),
        },
        index=dates,
    )


@pytest.fixture(scope="module")
def factory() -> StrategyFactory:
    return StrategyFactory()


@pytest.fixture(scope="module")
def engine() -> BacktestingEngine:
    return BacktestingEngine(initial_capital=100000)


def _sample_parameters(space: dict) -> dict:
    """Sample a concrete parameter set from a hyperopt space."""
    return {k: stochastic.sample(v) for k, v in space.items()}


# Test only a subset for quick validation
_STRATEGIES = StrategyFactory().get_all_strategies()[:3]


@pytest.mark.parametrize("strategy_name", _STRATEGIES)
def test_strategy_backtest(
    strategy_name: str,
    factory: StrategyFactory,
    engine: BacktestingEngine,
    test_data: pd.DataFrame,
) -> None:
    """Backtest strategy and verify basic metrics."""
    param_space = factory.get_parameter_space(strategy_name)
    parameters = _sample_parameters(param_space)
    strategy = factory.create_strategy(strategy_name, **parameters)
    engine.reset()
    result = engine.backtest_strategy(strategy, test_data, "TEST/USDT")

    assert result.total_trades > 0 or abs(result.total_return) <= 0.05, f"{strategy_name} produced no trades but had return {result.total_return}"
    assert -1 <= result.total_return <= 1, f"{strategy_name} unreasonable return"
    assert 0 <= result.win_rate <= 1, f"{strategy_name} invalid win rate"
    assert np.isfinite(result.sharpe_ratio), f"{strategy_name} sharpe not finite"

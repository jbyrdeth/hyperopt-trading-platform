"""
Multi-Exchange Data Fetcher for Trading Strategy Optimization

This module provides a robust data fetching system that can retrieve historical
OHLCV data from multiple cryptocurrency exchanges with intelligent caching,
data validation, and fallback mechanisms.
"""

import asyncio
import ccxt
import pandas as pd
import numpy as np
import sqlite3
import logging
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import aiohttp
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.logger import get_logger, PerformanceTimer, log_performance


@dataclass
class DataQualityMetrics:
    """Data quality metrics for validation."""
    total_points: int
    missing_points: int
    outliers: int
    gaps: int
    spike_count: int
    quality_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_points': self.total_points,
            'missing_points': self.missing_points,
            'outliers': self.outliers,
            'gaps': self.gaps,
            'spike_count': self.spike_count,
            'quality_score': self.quality_score
        }


@dataclass
class FetchResult:
    """Result of a data fetch operation."""
    success: bool
    data: Optional[pd.DataFrame]
    source: str
    symbol: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    fetch_duration: float
    quality_metrics: Optional[DataQualityMetrics]
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'source': self.source,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'fetch_duration': self.fetch_duration,
            'data_points': len(self.data) if self.data is not None else 0,
            'quality_metrics': self.quality_metrics.to_dict() if self.quality_metrics else None,
            'error_message': self.error_message
        }


class ExchangeConnector:
    """Base class for exchange connections with rate limiting and error handling."""
    
    def __init__(self, exchange_config: Dict[str, Any], logger: logging.Logger):
        self.config = exchange_config
        self.logger = logger
        self.exchange = None
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit = exchange_config.get('rate_limit', 10)  # requests per second
        self.timeout = exchange_config.get('timeout', 30000)
        self.max_retries = 3
        
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize the CCXT exchange instance."""
        try:
            exchange_name = self.config['name'].lower().replace(' ', '')
            
            # Map exchange names to CCXT classes
            exchange_mapping = {
                'kucoin': ccxt.kucoin,
                'binance': ccxt.binance,
                'coinbasepro': ccxt.coinbase,  # Updated to use 'coinbase' instead of 'coinbasepro'
                'coinbase': ccxt.coinbase  # Alias
            }
            
            if exchange_name not in exchange_mapping:
                raise ValueError(f"Unsupported exchange: {exchange_name}")
            
            # Initialize exchange
            exchange_class = exchange_mapping[exchange_name]
            self.exchange = exchange_class({
                'apiKey': self.config.get('api_key'),
                'secret': self.config.get('secret'),
                'password': self.config.get('passphrase'),  # For Coinbase Pro
                'sandbox': self.config.get('sandbox', False),
                'timeout': self.timeout,
                'enableRateLimit': True,
                'rateLimit': int(1000 / self.rate_limit),  # Convert to milliseconds
            })
            
            self.logger.info(f"Initialized {self.config['name']} exchange connector")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.config['name']}: {e}")
            raise
    
    async def _rate_limit_check(self):
        """Ensure we don't exceed rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch OHLCV data from the exchange.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (1h, 4h, 1d, etc.)
            since: Start timestamp in milliseconds
            limit: Maximum number of candles
            
        Returns:
            Tuple of (success, dataframe, error_message)
        """
        await self._rate_limit_check()
        
        for attempt in range(self.max_retries):
            try:
                # Convert symbol to exchange format
                exchange_symbol = self.config['assets'].get(symbol.split('/')[0], symbol)
                
                # Fetch data
                ohlcv = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.exchange.fetch_ohlcv,
                    exchange_symbol,
                    timeframe,
                    since,
                    limit
                )
                
                if not ohlcv:
                    return False, None, "No data returned from exchange"
                
                # Convert to DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Ensure numeric types
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_columns] = df[numeric_columns].astype(float)
                
                self.logger.debug(
                    f"Fetched {len(df)} candles for {symbol} {timeframe} from {self.config['name']}"
                )
                
                return True, df, None
                
            except ccxt.NetworkError as e:
                self.logger.warning(f"Network error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
                
            except ccxt.RateLimitExceeded as e:
                self.logger.warning(f"Rate limit exceeded on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(60)  # Wait 1 minute
                continue
                
            except Exception as e:
                error_msg = f"Error fetching data from {self.config['name']}: {e}"
                self.logger.error(error_msg)
                return False, None, error_msg
        
        return False, None, f"Failed after {self.max_retries} attempts"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            'exchange': self.config['name'],
            'request_count': self.request_count,
            'rate_limit': self.rate_limit,
            'last_request': self.last_request_time
        }


class DataCache:
    """SQLite-based caching system for OHLCV data."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "ohlcv_cache.db"
        self.logger = get_logger()
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    quality_score REAL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    UNIQUE(symbol, timeframe, exchange, start_time, end_time)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_lookup 
                ON ohlcv_data(symbol, timeframe, exchange, start_time, end_time)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at 
                ON ohlcv_data(expires_at)
            """)
    
    def _generate_cache_key(
        self,
        symbol: str,
        timeframe: str,
        exchange: str,
        start_time: datetime,
        end_time: datetime
    ) -> str:
        """Generate a unique cache key."""
        key_string = f"{symbol}_{timeframe}_{exchange}_{start_time}_{end_time}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cached_data(
        self,
        symbol: str,
        timeframe: str,
        exchange: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[pd.DataFrame]:
        """Retrieve cached data if available and not expired."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT data_json, quality_score FROM ohlcv_data
                    WHERE symbol = ? AND timeframe = ? AND exchange = ?
                    AND start_time = ? AND end_time = ?
                    AND expires_at > ?
                """, (
                    symbol, timeframe, exchange,
                    start_time.isoformat(), end_time.isoformat(),
                    datetime.now().isoformat()
                ))
                
                row = cursor.fetchone()
                if row:
                    data_json, quality_score = row
                    df = pd.read_json(data_json, orient='index')
                    df.index = pd.to_datetime(df.index)
                    
                    self.logger.debug(
                        f"Cache hit for {symbol} {timeframe} {exchange} "
                        f"(quality: {quality_score:.3f})"
                    )
                    return df
                
        except Exception as e:
            self.logger.warning(f"Error retrieving cached data: {e}")
        
        return None
    
    def cache_data(
        self,
        symbol: str,
        timeframe: str,
        exchange: str,
        start_time: datetime,
        end_time: datetime,
        data: pd.DataFrame,
        quality_score: float,
        expiry_hours: int = 24
    ):
        """Cache OHLCV data."""
        try:
            data_json = data.to_json(orient='index', date_format='iso')
            data_hash = hashlib.md5(data_json.encode()).hexdigest()
            expires_at = datetime.now() + timedelta(hours=expiry_hours)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO ohlcv_data
                    (symbol, timeframe, exchange, start_time, end_time, 
                     data_hash, data_json, quality_score, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, timeframe, exchange,
                    start_time.isoformat(), end_time.isoformat(),
                    data_hash, data_json, quality_score,
                    datetime.now().isoformat(), expires_at.isoformat()
                ))
            
            self.logger.debug(f"Cached data for {symbol} {timeframe} {exchange}")
            
        except Exception as e:
            self.logger.warning(f"Error caching data: {e}")
    
    def cleanup_expired(self):
        """Remove expired cache entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM ohlcv_data WHERE expires_at < ?
                """, (datetime.now().isoformat(),))
                
                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} expired cache entries")
                    
        except Exception as e:
            self.logger.warning(f"Error cleaning up cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM ohlcv_data")
                total_entries = cursor.fetchone()[0]
                
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM ohlcv_data WHERE expires_at > ?
                """, (datetime.now().isoformat(),))
                valid_entries = cursor.fetchone()[0]
                
                cursor = conn.execute("""
                    SELECT AVG(quality_score) FROM ohlcv_data WHERE expires_at > ?
                """, (datetime.now().isoformat(),))
                avg_quality = cursor.fetchone()[0] or 0
                
                return {
                    'total_entries': total_entries,
                    'valid_entries': valid_entries,
                    'expired_entries': total_entries - valid_entries,
                    'average_quality': avg_quality,
                    'cache_file_size': self.db_path.stat().st_size if self.db_path.exists() else 0
                }
                
        except Exception as e:
            self.logger.warning(f"Error getting cache stats: {e}")
            return {}


class DataValidator:
    """Data quality validation and cleaning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger()
    
    def validate_data(self, data: pd.DataFrame, symbol: str) -> DataQualityMetrics:
        """
        Comprehensive data validation.
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading pair symbol
            
        Returns:
            DataQualityMetrics object
        """
        total_points = len(data)
        issues = {
            'missing_points': 0,
            'outliers': 0,
            'gaps': 0,
            'spike_count': 0
        }
        
        if total_points == 0:
            return DataQualityMetrics(0, 0, 0, 0, 0, 0.0)
        
        # Check for missing data
        missing_mask = data.isnull().any(axis=1)
        issues['missing_points'] = missing_mask.sum()
        
        # Check OHLC consistency
        ohlc_issues = (
            (data['open'] > data['high']) |
            (data['open'] < data['low']) |
            (data['close'] > data['high']) |
            (data['close'] < data['low']) |
            (data['high'] < data['low'])
        )
        issues['outliers'] += ohlc_issues.sum()
        
        # Check for price spikes
        if self.config.get('spike_detection', {}).get('enabled', True):
            spike_threshold = self.config['spike_detection'].get('threshold', 0.1)
            price_changes = data['close'].pct_change().abs()
            spikes = price_changes > spike_threshold
            issues['spike_count'] = spikes.sum()
        
        # Check for time gaps
        if self.config.get('gap_handling', {}).get('enabled', True):
            max_gap_hours = self.config['gap_handling'].get('max_gap_hours', 2)
            time_diffs = data.index.to_series().diff()
            expected_interval = time_diffs.median()
            large_gaps = time_diffs > expected_interval * 2
            issues['gaps'] = large_gaps.sum()
        
        # Check for volume anomalies
        if 'volume' in data.columns:
            zero_volume = (data['volume'] <= 0).sum()
            issues['outliers'] += zero_volume
        
        # Calculate quality score (0-1, higher is better)
        total_issues = sum(issues.values())
        quality_score = max(0, 1 - (total_issues / total_points))
        
        metrics = DataQualityMetrics(
            total_points=total_points,
            missing_points=issues['missing_points'],
            outliers=issues['outliers'],
            gaps=issues['gaps'],
            spike_count=issues['spike_count'],
            quality_score=quality_score
        )
        
        self.logger.debug(
            f"Data validation for {symbol}: "
            f"Quality={quality_score:.3f}, Issues={total_issues}/{total_points}"
        )
        
        return metrics
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and fix data issues.
        
        Args:
            data: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()
        
        # Remove rows with missing critical data
        cleaned_data = cleaned_data.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Fix OHLC inconsistencies
        cleaned_data['high'] = cleaned_data[['open', 'high', 'close']].max(axis=1)
        cleaned_data['low'] = cleaned_data[['open', 'low', 'close']].min(axis=1)
        
        # Handle zero or negative volumes
        if 'volume' in cleaned_data.columns:
            cleaned_data['volume'] = cleaned_data['volume'].clip(lower=0)
            # Handle zero volume (replace with forward fill)
            cleaned_data['volume'] = cleaned_data['volume'].replace(0, np.nan).ffill()
        
        # Handle price spikes if configured
        if self.config.get('spike_detection', {}).get('action') == 'remove':
            spike_threshold = self.config['spike_detection'].get('threshold', 0.1)
            price_changes = cleaned_data['close'].pct_change().abs()
            spike_mask = price_changes <= spike_threshold
            cleaned_data = cleaned_data[spike_mask]
        
        # Sort by timestamp to ensure proper order
        cleaned_data = cleaned_data.sort_index()
        
        self.logger.debug(f"Cleaned data: {len(data)} -> {len(cleaned_data)} rows")
        
        return cleaned_data


class DataFetcher:
    """
    Main data fetching orchestrator with multi-exchange support.
    
    Features:
    - Multi-exchange data fetching with fallback
    - Intelligent caching
    - Data validation and cleaning
    - Concurrent fetching for performance
    - Comprehensive error handling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data fetcher.
        
        Args:
            config: Exchange configuration dictionary
        """
        self.config = config
        self.logger = get_logger()
        
        # Initialize components
        self.cache = DataCache(config.get('caching', {}).get('cache_directory', 'data/cache'))
        self.validator = DataValidator(config.get('data_quality', {}))
        
        # Initialize exchange connectors
        self.exchanges = {}
        self._initialize_exchanges()
        
        # Performance tracking
        self.fetch_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'failed_requests': 0,
            'total_fetch_time': 0
        }
        
        self.logger.info(f"DataFetcher initialized with {len(self.exchanges)} exchanges")
    
    def _initialize_exchanges(self):
        """Initialize exchange connectors based on configuration."""
        exchanges_config = self.config.get('exchanges', {})
        
        for exchange_name, exchange_config in exchanges_config.items():
            if exchange_config.get('enabled', False):
                try:
                    connector = ExchangeConnector(exchange_config, self.logger)
                    self.exchanges[exchange_name] = connector
                    self.logger.info(f"Initialized {exchange_name} connector")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {exchange_name}: {e}")
    
    def get_exchange_priority_order(self) -> List[str]:
        """Get exchanges ordered by priority."""
        exchange_priorities = []
        for name, connector in self.exchanges.items():
            priority = connector.config.get('priority', 999)
            exchange_priorities.append((priority, name))
        
        exchange_priorities.sort()
        return [name for _, name in exchange_priorities]
    
    @log_performance("fetch_data")
    async def fetch_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        preferred_exchange: Optional[str] = None
    ) -> FetchResult:
        """
        Fetch OHLCV data with fallback logic.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (1h, 4h, 1d, etc.)
            start_date: Start date for data
            end_date: End date for data
            preferred_exchange: Preferred exchange name
            
        Returns:
            FetchResult object with data and metadata
        """
        start_time = time.time()
        self.fetch_stats['total_requests'] += 1
        
        # Determine exchange order
        if preferred_exchange and preferred_exchange in self.exchanges:
            exchange_order = [preferred_exchange] + [
                ex for ex in self.get_exchange_priority_order() 
                if ex != preferred_exchange
            ]
        else:
            exchange_order = self.get_exchange_priority_order()
        
        # Try cache first
        for exchange_name in exchange_order:
            cached_data = self.cache.get_cached_data(
                symbol, timeframe, exchange_name, start_date, end_date
            )
            if cached_data is not None:
                self.fetch_stats['cache_hits'] += 1
                quality_metrics = self.validator.validate_data(cached_data, symbol)
                
                return FetchResult(
                    success=True,
                    data=cached_data,
                    source=f"{exchange_name} (cached)",
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_date,
                    end_time=end_date,
                    fetch_duration=time.time() - start_time,
                    quality_metrics=quality_metrics
                )
        
        # Cache miss - fetch from exchanges
        self.fetch_stats['cache_misses'] += 1
        
        for exchange_name in exchange_order:
            if exchange_name not in self.exchanges:
                continue
            
            try:
                connector = self.exchanges[exchange_name]
                
                # Convert datetime to timestamp
                since = int(start_date.timestamp() * 1000)
                
                # Fetch data
                success, raw_data, error_msg = await connector.fetch_ohlcv(
                    symbol, timeframe, since
                )
                
                if success and raw_data is not None:
                    # Filter data to requested range
                    mask = (raw_data.index >= start_date) & (raw_data.index <= end_date)
                    filtered_data = raw_data[mask]
                    
                    if len(filtered_data) == 0:
                        self.logger.warning(f"No data in requested range from {exchange_name}")
                        continue
                    
                    # Validate and clean data
                    quality_metrics = self.validator.validate_data(filtered_data, symbol)
                    cleaned_data = self.validator.clean_data(filtered_data)
                    
                    # Cache the data
                    cache_expiry = self.config.get('caching', {}).get('expiry_rules', {}).get(timeframe, 3600) // 3600
                    self.cache.cache_data(
                        symbol, timeframe, exchange_name,
                        start_date, end_date, cleaned_data,
                        quality_metrics.quality_score, cache_expiry
                    )
                    
                    fetch_duration = time.time() - start_time
                    self.fetch_stats['total_fetch_time'] += fetch_duration
                    
                    return FetchResult(
                        success=True,
                        data=cleaned_data,
                        source=exchange_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        start_time=start_date,
                        end_time=end_date,
                        fetch_duration=fetch_duration,
                        quality_metrics=quality_metrics
                    )
                
                else:
                    self.logger.warning(f"Failed to fetch from {exchange_name}: {error_msg}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching from {exchange_name}: {e}")
                continue
        
        # All exchanges failed
        self.fetch_stats['failed_requests'] += 1
        
        return FetchResult(
            success=False,
            data=None,
            source="none",
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_date,
            end_time=end_date,
            fetch_duration=time.time() - start_time,
            quality_metrics=None,
            error_message="All exchanges failed to provide data"
        )
    
    async def fetch_multiple_assets(
        self,
        symbols: List[str],
        timeframes: List[str],
        start_date: datetime = None,
        end_date: datetime = None,
        days_back: int = 365
    ) -> Dict[str, Dict[str, FetchResult]]:
        """
        Fetch data for multiple assets and timeframes concurrently.
        
        Args:
            symbols: List of trading pair symbols
            timeframes: List of timeframes
            start_date: Start date (optional)
            end_date: End date (optional)
            days_back: Days back from now if dates not specified
            
        Returns:
            Nested dictionary: {symbol: {timeframe: FetchResult}}
        """
        # Set default date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=days_back)
        
        self.logger.info(
            f"Fetching data for {len(symbols)} symbols, {len(timeframes)} timeframes "
            f"from {start_date.date()} to {end_date.date()}"
        )
        
        # Create tasks for concurrent fetching
        tasks = []
        task_keys = []
        for symbol in symbols:
            for timeframe in timeframes:
                task = asyncio.create_task(
                    self.fetch_data(symbol, timeframe, start_date, end_date)
                )
                tasks.append(task)
                task_keys.append((symbol, timeframe))

        # Execute tasks concurrently
        results = {}
        completed_tasks = 0
        total_tasks = len(tasks)

        try:
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            self.logger.error(f"Unexpected error during batch fetch: {e}")
            gathered = []

        for (symbol, timeframe), result in zip(task_keys, gathered):
            if isinstance(result, Exception):
                self.logger.error(f"Error fetching {symbol} {timeframe}: {result}")
                fetch_result = FetchResult(
                    success=False,
                    data=None,
                    source="error",
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_date,
                    end_time=end_date,
                    fetch_duration=0,
                    quality_metrics=None,
                    error_message=str(result)
                )
            else:
                fetch_result = result

            if symbol not in results:
                results[symbol] = {}
            results[symbol][timeframe] = fetch_result

            completed_tasks += 1

            if completed_tasks % 10 == 0 or completed_tasks == total_tasks:
                self.logger.info(
                    f"Completed {completed_tasks}/{total_tasks} fetch tasks"
                )
        
        # Log summary
        successful_fetches = sum(
            1 for symbol_results in results.values()
            for result in symbol_results.values()
            if result.success
        )
        
        self.logger.info(
            f"Fetch completed: {successful_fetches}/{total_tasks} successful, "
            f"Cache hits: {self.fetch_stats['cache_hits']}, "
            f"Cache misses: {self.fetch_stats['cache_misses']}"
        )
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive fetcher statistics."""
        exchange_stats = {
            name: connector.get_stats()
            for name, connector in self.exchanges.items()
        }
        
        cache_stats = self.cache.get_cache_stats()
        
        return {
            'fetch_stats': self.fetch_stats,
            'exchange_stats': exchange_stats,
            'cache_stats': cache_stats,
            'active_exchanges': list(self.exchanges.keys())
        }
    
    def cleanup_cache(self):
        """Clean up expired cache entries."""
        self.cache.cleanup_expired()
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all exchange connections."""
        health_status = {}
        
        for exchange_name, connector in self.exchanges.items():
            try:
                # Try to fetch a small amount of data
                success, _, _ = await connector.fetch_ohlcv('BTC/USDT', '1d', limit=1)
                health_status[exchange_name] = success
            except Exception as e:
                self.logger.warning(f"Health check failed for {exchange_name}: {e}")
                health_status[exchange_name] = False
        
        return health_status 
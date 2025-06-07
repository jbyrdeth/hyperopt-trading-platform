#!/usr/bin/env python3
"""
Test script for the Multi-Exchange Data Fetcher

This script tests the data fetching functionality without requiring API keys.
It demonstrates the system's ability to handle errors gracefully and shows
the complete data fetching workflow.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging
from src.data.data_fetcher import DataFetcher


async def test_data_fetcher():
    """Test the data fetcher functionality."""
    
    # Setup logging
    logger = setup_logging(level="INFO", console_output=True)
    logger.info("Starting Data Fetcher Test")
    
    try:
        # Load configuration
        config_manager = ConfigManager("config")
        exchange_config = config_manager.get_config("exchanges")
        
        # Initialize data fetcher
        logger.info("Initializing DataFetcher...")
        data_fetcher = DataFetcher(exchange_config)
        
        # Test health check
        logger.info("Performing health check...")
        health_status = await data_fetcher.health_check()
        logger.info(f"Health check results: {health_status}")
        
        # Test single asset fetch
        logger.info("Testing single asset data fetch...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last 7 days
        
        result = await data_fetcher.fetch_data(
            symbol="BTC/USDT",
            timeframe="1d",
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"Fetch result: {result.to_dict()}")
        
        if result.success:
            logger.info(f"Successfully fetched {len(result.data)} data points")
            logger.info(f"Data quality score: {result.quality_metrics.quality_score:.3f}")
            logger.info(f"Data source: {result.source}")
            
            # Show sample data
            if len(result.data) > 0:
                logger.info("Sample data (first 3 rows):")
                print(result.data.head(3))
        else:
            logger.warning(f"Fetch failed: {result.error_message}")
        
        # Test multiple assets fetch
        logger.info("Testing multiple assets fetch...")
        symbols = ["BTC/USDT", "ETH/USDT"]
        timeframes = ["1d"]
        
        multi_results = await data_fetcher.fetch_multiple_assets(
            symbols=symbols,
            timeframes=timeframes,
            days_back=3  # Last 3 days for faster testing
        )
        
        # Show results summary
        for symbol, timeframe_results in multi_results.items():
            for timeframe, fetch_result in timeframe_results.items():
                logger.info(
                    f"{symbol} {timeframe}: "
                    f"Success={fetch_result.success}, "
                    f"Points={len(fetch_result.data) if fetch_result.data is not None else 0}, "
                    f"Source={fetch_result.source}"
                )
        
        # Show statistics
        stats = data_fetcher.get_stats()
        logger.info("Data Fetcher Statistics:")
        logger.info(f"  Total requests: {stats['fetch_stats']['total_requests']}")
        logger.info(f"  Cache hits: {stats['fetch_stats']['cache_hits']}")
        logger.info(f"  Cache misses: {stats['fetch_stats']['cache_misses']}")
        logger.info(f"  Failed requests: {stats['fetch_stats']['failed_requests']}")
        logger.info(f"  Active exchanges: {stats['active_exchanges']}")
        
        # Test cache cleanup
        logger.info("Testing cache cleanup...")
        data_fetcher.cleanup_cache()
        
        logger.info("Data Fetcher test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Main test function."""
    print("="*60)
    print("TRADING OPTIMIZATION SYSTEM - DATA FETCHER TEST")
    print("="*60)
    
    # Run the async test
    success = asyncio.run(test_data_fetcher())
    
    if success:
        print("\n✅ All tests passed! Data fetcher is working correctly.")
        print("\nNext steps:")
        print("1. Add API keys to config/exchanges.yaml for live data")
        print("2. Test with real exchange connections")
        print("3. Proceed to implement strategy framework")
    else:
        print("\n❌ Tests failed. Check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main() 
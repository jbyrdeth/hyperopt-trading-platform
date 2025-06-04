"""
Strategies Router

Handles strategy-related endpoints for the trading strategy optimization API.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
import logging

from models import (
    StrategyListResponse, StrategyInfo, StrategyParameter, StrategyType,
    BaseResponse, Asset, TimeFrame
)
from auth import verify_api_key, require_permission

# CRITICAL FIX: Import the actual strategy factory instead of using mock data
try:
    from optimization.strategy_factory import StrategyFactory
except ImportError:
    from ..optimization.strategy_factory import StrategyFactory

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize strategy factory
strategy_factory = StrategyFactory()

def get_strategy_type_from_category(category: str) -> StrategyType:
    """Convert strategy factory category to API StrategyType enum."""
    category_mapping = {
        "trend_following": StrategyType.TREND_FOLLOWING,
        "mean_reversion": StrategyType.MEAN_REVERSION,
        "momentum": StrategyType.MOMENTUM,
        "volume": StrategyType.VOLUME,
        "volatility": StrategyType.VOLATILITY,
        "pattern_recognition": StrategyType.PATTERN,
        "multi_timeframe": StrategyType.MULTI_TIMEFRAME
    }
    return category_mapping.get(category, StrategyType.TREND_FOLLOWING)

def convert_parameter_space_to_api_format(param_space: Dict[str, Any]) -> Dict[str, StrategyParameter]:
    """Convert strategy factory parameter space to API format."""
    api_params = {}
    
    for param_name, param_config in param_space.items():
        # Handle different parameter space formats
        if isinstance(param_config, dict):
            if 'low' in param_config and 'high' in param_config:
                # Hyperopt-style parameter space
                param_type = "float" if isinstance(param_config.get('low'), float) else "int"
                api_params[param_name] = StrategyParameter(
                    name=param_name,
                    value=param_config.get('low', 0),  # Default to low value
                    type=param_type,
                    min_value=param_config.get('low'),
                    max_value=param_config.get('high'),
                    description=param_config.get('description', f"{param_name} parameter")
                )
            elif 'choices' in param_config:
                # Choice parameter
                api_params[param_name] = StrategyParameter(
                    name=param_name,
                    value=param_config['choices'][0],  # Default to first choice
                    type="str",
                    min_value=None,
                    max_value=None,
                    description=param_config.get('description', f"{param_name} parameter"),
                    choices=param_config['choices']
                )
        else:
            # Simple value - create basic parameter
            param_type = "float" if isinstance(param_config, float) else "int" if isinstance(param_config, int) else "str"
            api_params[param_name] = StrategyParameter(
                name=param_name,
                value=param_config,
                type=param_type,
                min_value=None,
                max_value=None,
                description=f"{param_name} parameter"
            )
    
    return api_params

def get_strategy_info_from_factory(strategy_name: str) -> Dict[str, Any]:
    """Get strategy information from the factory and convert to API format."""
    try:
        # Get strategy class and parameter space
        strategy_class = strategy_factory.registry.get_strategy_class(strategy_name)
        param_space = strategy_factory.get_parameter_space(strategy_name)
        strategy_info = strategy_factory.registry.get_strategy_info(strategy_name)
        
        # Convert parameters to API format
        api_parameters = convert_parameter_space_to_api_format(param_space)
        
        # Get category and convert to StrategyType
        category = strategy_info.get('category', 'trend_following')
        strategy_type = get_strategy_type_from_category(category)
        
        # Create strategy info
        return {
            "name": strategy_name,
            "description": strategy_info.get('description', f"{strategy_name} trading strategy"),
            "category": strategy_type,
            "parameters": api_parameters,
            "default_timeframe": TimeFrame.H4,  # Default timeframe
            "recommended_assets": [Asset.BTC, Asset.ETH],  # Default assets
            "risk_level": strategy_info.get('risk_level', 'Medium'),
            "complexity_score": strategy_info.get('complexity_score', 5.0)
        }
        
    except Exception as e:
        logger.error(f"Error getting strategy info for {strategy_name}: {e}")
        raise

def get_all_strategies_from_factory() -> Dict[str, Dict[str, Any]]:
    """Get all strategies from the factory in API format."""
    strategies = {}
    
    for strategy_name in strategy_factory.get_all_strategies():
        try:
            strategies[strategy_name] = get_strategy_info_from_factory(strategy_name)
        except Exception as e:
            logger.warning(f"Skipping strategy {strategy_name}: {e}")
            continue
    
    return strategies

@router.get("", response_model=StrategyListResponse)
async def list_strategies(
    category: Optional[StrategyType] = Query(None, description="Filter by strategy category"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level (Low, Medium, High)"),
    complexity_min: Optional[float] = Query(None, ge=1, le=10, description="Minimum complexity score"),
    complexity_max: Optional[float] = Query(None, ge=1, le=10, description="Maximum complexity score"),
    search: Optional[str] = Query(None, description="Search strategy names and descriptions"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of strategies to return"),
    offset: int = Query(0, ge=0, description="Number of strategies to skip"),
    key_info: Dict = Depends(verify_api_key)
):
    """
    List available trading strategies.
    
    Returns a paginated list of trading strategies with filtering options.
    Supports filtering by category, risk level, complexity, and text search.
    
    **NEW**: Now returns all 24 strategies from the strategy factory instead of mock data.
    """
    try:
        logger.info(f"Listing strategies: category={category}, risk_level={risk_level}")
        
        # Get all strategies from factory
        all_strategies = get_all_strategies_from_factory()
        strategies = list(all_strategies.values())
        
        logger.info(f"Loaded {len(strategies)} strategies from factory")
        
        # Apply filters
        if category:
            strategies = [s for s in strategies if s["category"] == category]
        
        if risk_level:
            strategies = [s for s in strategies if s["risk_level"] == risk_level]
        
        if complexity_min is not None:
            strategies = [s for s in strategies if s.get("complexity_score", 0) >= complexity_min]
        
        if complexity_max is not None:
            strategies = [s for s in strategies if s.get("complexity_score", 10) <= complexity_max]
        
        if search:
            search_lower = search.lower()
            strategies = [
                s for s in strategies 
                if search_lower in s["name"].lower() or search_lower in s["description"].lower()
            ]
        
        # Apply pagination
        total_count = len(strategies)
        strategies = strategies[offset:offset + limit]
        
        # Convert to response models
        strategy_models = [StrategyInfo(**strategy) for strategy in strategies]
        
        # Calculate category counts from all strategies
        categories = {}
        for strategy in all_strategies.values():
            cat = strategy["category"].value
            categories[cat] = categories.get(cat, 0) + 1
        
        logger.info(f"Returning {len(strategy_models)} strategies (filtered from {total_count})")
        
        return StrategyListResponse(
            strategies=strategy_models,
            total_count=total_count,
            categories=categories
        )
        
    except Exception as e:
        logger.error(f"Error listing strategies: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve strategies: {str(e)}"
        )


@router.get("/{strategy_name}", response_model=StrategyInfo)
async def get_strategy(
    strategy_name: str,
    key_info: Dict = Depends(verify_api_key)
):
    """
    Get detailed information about a specific strategy.
    
    Returns comprehensive details including parameters, default settings,
    risk level, and complexity analysis.
    
    **NEW**: Now supports all 24 strategies from the strategy factory.
    """
    try:
        logger.info(f"Getting strategy details: {strategy_name}")
        
        # Check if strategy exists in factory
        if strategy_name not in strategy_factory.get_all_strategies():
            raise HTTPException(
                status_code=404,
                detail=f"Strategy '{strategy_name}' not found. Available strategies: {', '.join(strategy_factory.get_all_strategies())}"
            )
        
        # Get strategy info from factory
        strategy_data = get_strategy_info_from_factory(strategy_name)
        return StrategyInfo(**strategy_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving strategy {strategy_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve strategy: {str(e)}"
        )


@router.post("/{strategy_name}/validate", response_model=BaseResponse)
async def validate_strategy_parameters(
    strategy_name: str,
    parameters: Dict[str, Any],
    key_info: Dict = Depends(require_permission("read"))
):
    """
    Validate strategy parameters.
    
    Checks if the provided parameters are valid for the specified strategy,
    including type validation, range checking, and parameter relationships.
    
    **NEW**: Now validates against actual strategy factory parameter spaces.
    """
    try:
        logger.info(f"Validating parameters for strategy: {strategy_name}")
        
        # Check if strategy exists
        if strategy_name not in strategy_factory.get_all_strategies():
            raise HTTPException(
                status_code=404,
                detail=f"Strategy '{strategy_name}' not found"
            )
        
        # Use strategy factory validation
        is_valid = strategy_factory.validate_strategy_parameters(strategy_name, parameters)
        
        if not is_valid:
            raise HTTPException(
                status_code=422,
                detail={
                    "error_code": "PARAMETER_VALIDATION_FAILED",
                    "error_message": "Parameter validation failed",
                    "validation_errors": ["One or more parameters are invalid"]
                }
            )
        
        return BaseResponse(success=True)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating parameters for {strategy_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate parameters: {str(e)}"
        )


@router.get("/categories/summary", response_model=Dict[str, Any])
async def get_strategy_categories(
    key_info: Dict = Depends(verify_api_key)
):
    """
    Get summary of strategy categories.
    
    Returns information about available strategy categories including
    counts, descriptions, and recommended use cases.
    
    **NEW**: Now returns actual category data from the strategy factory.
    """
    try:
        # Get strategy summary from factory
        factory_summary = strategy_factory.get_strategy_summary()
        
        # Convert to API format
        categories = {}
        for category, info in factory_summary["categories"].items():
            api_category = get_strategy_type_from_category(category).value
            categories[api_category] = {
                "name": api_category,
                "count": info["count"],
                "examples": info["strategies"][:3],  # Limit to 3 examples
                "description": get_category_description(api_category),
                "risk_levels": ["Low", "Medium", "High"],  # Default risk levels
                "avg_complexity": 5.0  # Default complexity
            }
        
        return {
            "categories": categories,
            "total_strategies": factory_summary["total_strategies"],
            "total_categories": len(categories)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving category summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve category summary: {str(e)}"
        )


def get_category_description(category: str) -> str:
    """Get description for a strategy category."""
    descriptions = {
        "trend_following": "Strategies that identify and follow market trends using moving averages and trend indicators",
        "mean_reversion": "Strategies that bet on price reversals to the mean using oscillators and statistical measures",
        "momentum": "Strategies that capitalize on price momentum and acceleration using momentum indicators",
        "volume": "Strategies that incorporate volume analysis to confirm price movements and identify accumulation/distribution",
        "volatility": "Strategies that trade based on volatility patterns and volatility breakouts/contractions",
        "pattern": "Strategies that recognize and trade chart patterns and technical formations",
        "multi_timeframe": "Strategies that analyze multiple timeframes for enhanced signal confirmation"
    }
    return descriptions.get(category, "Strategy category description not available")


# REMOVED: load_strategies_from_factory() function - no longer needed as we're directly integrated 
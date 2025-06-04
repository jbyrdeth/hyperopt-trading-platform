"""
Parameter Translator

This module provides functionality to translate Python strategy parameters
into Pine Script v5 input parameter syntax with proper types and constraints.
"""

from typing import Dict, List, Any, Union, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import logging

try:
    from ..utils.logger import get_logger
except ImportError:
    from src.utils.logger import get_logger


class PineScriptType(Enum):
    """Pine Script input types."""
    INT = "input.int"
    FLOAT = "input.float"
    BOOL = "input.bool"
    STRING = "input.string"
    SOURCE = "input.source"
    TIMEFRAME = "input.timeframe"
    SESSION = "input.session"
    COLOR = "input.color"


@dataclass
class PineScriptParameter:
    """Represents a Pine Script input parameter."""
    
    name: str
    param_type: PineScriptType
    default_value: Any
    title: str
    tooltip: str = ""
    group: str = "Strategy Parameters"
    minval: Optional[float] = None
    maxval: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None
    confirm: bool = False
    
    def to_pine_script(self) -> str:
        """Convert to Pine Script input syntax."""
        
        # Build the input line
        parts = [f"{self.param_type.value}("]
        parts.append(f"{self._format_value(self.default_value)}")
        parts.append(f'"{self.title}"')
        
        # Add optional parameters
        if self.tooltip:
            parts.append(f'tooltip="{self.tooltip}"')
        
        if self.minval is not None:
            parts.append(f"minval={self.minval}")
            
        if self.maxval is not None:
            parts.append(f"maxval={self.maxval}")
            
        if self.step is not None:
            parts.append(f"step={self.step}")
            
        if self.options:
            options_str = ', '.join(f'"{opt}"' for opt in self.options)
            parts.append(f"options=[{options_str}]")
            
        if self.group:
            parts.append(f'group="{self.group}"')
            
        if self.confirm:
            parts.append("confirm=true")
        
        # Join and format - fix the comma issue
        param_content = ", ".join(parts[1:])  # Skip the opening parenthesis
        param_line = f"{parts[0]}{param_content})"
        return f"{self.name} = {param_line}"
    
    def _format_value(self, value: Any) -> str:
        """Format a value for Pine Script."""
        if isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            return str(value)


class ParameterTranslator:
    """Translates Python strategy parameters to Pine Script inputs."""
    
    def __init__(self):
        self.logger = get_logger("parameter_translator")
        
        # Parameter name mapping and type detection rules
        self.parameter_rules = {
            # Moving Average parameters
            "fast_period": {
                "type": PineScriptType.INT,
                "title": "Fast MA Period",
                "group": "Moving Average",
                "minval": 1,
                "maxval": 200,
                "step": 1,
                "tooltip": "Period for fast moving average"
            },
            "slow_period": {
                "type": PineScriptType.INT,
                "title": "Slow MA Period", 
                "group": "Moving Average",
                "minval": 1,
                "maxval": 500,
                "step": 1,
                "tooltip": "Period for slow moving average"
            },
            "ma_type": {
                "type": PineScriptType.STRING,
                "title": "MA Type",
                "group": "Moving Average",
                "options": ["SMA", "EMA", "WMA", "RMA"],
                "tooltip": "Type of moving average to use"
            },
            
            # RSI parameters
            "rsi_period": {
                "type": PineScriptType.INT,
                "title": "RSI Period",
                "group": "RSI",
                "minval": 1,
                "maxval": 100,
                "step": 1,
                "tooltip": "Period for RSI calculation"
            },
            "rsi_overbought": {
                "type": PineScriptType.FLOAT,
                "title": "RSI Overbought",
                "group": "RSI",
                "minval": 50.0,
                "maxval": 100.0,
                "step": 1.0,
                "tooltip": "RSI overbought threshold"
            },
            "rsi_oversold": {
                "type": PineScriptType.FLOAT,
                "title": "RSI Oversold",
                "group": "RSI",
                "minval": 0.0,
                "maxval": 50.0,
                "step": 1.0,
                "tooltip": "RSI oversold threshold"
            },
            "confirmation_period": {
                "type": PineScriptType.INT,
                "title": "Confirmation Period",
                "group": "RSI",
                "minval": 1,
                "maxval": 10,
                "step": 1,
                "tooltip": "Period for signal confirmation"
            },
            
            # MACD parameters
            "macd_fast": {
                "type": PineScriptType.INT,
                "title": "MACD Fast",
                "group": "MACD",
                "minval": 1,
                "maxval": 50,
                "step": 1,
                "tooltip": "Fast EMA period for MACD"
            },
            "macd_slow": {
                "type": PineScriptType.INT,
                "title": "MACD Slow",
                "group": "MACD",
                "minval": 1,
                "maxval": 100,
                "step": 1,
                "tooltip": "Slow EMA period for MACD"
            },
            "macd_signal": {
                "type": PineScriptType.INT,
                "title": "MACD Signal",
                "group": "MACD",
                "minval": 1,
                "maxval": 50,
                "step": 1,
                "tooltip": "Signal line period for MACD"
            },
            
            # Bollinger Bands parameters
            "bb_period": {
                "type": PineScriptType.INT,
                "title": "BB Period",
                "group": "Bollinger Bands",
                "minval": 1,
                "maxval": 100,
                "step": 1,
                "tooltip": "Period for Bollinger Bands"
            },
            "bb_std": {
                "type": PineScriptType.FLOAT,
                "title": "BB Standard Deviation",
                "group": "Bollinger Bands",
                "minval": 0.1,
                "maxval": 5.0,
                "step": 0.1,
                "tooltip": "Standard deviation multiplier"
            },
            
            # Generic threshold parameters
            "signal_threshold": {
                "type": PineScriptType.FLOAT,
                "title": "Signal Threshold",
                "group": "Signals",
                "minval": 0.0,
                "maxval": 1.0,
                "step": 0.01,
                "tooltip": "Minimum signal strength threshold"
            },
            "volume_threshold": {
                "type": PineScriptType.FLOAT,
                "title": "Volume Threshold",
                "group": "Volume",
                "minval": 0.1,
                "maxval": 10.0,
                "step": 0.1,
                "tooltip": "Volume multiplier threshold"
            },
            "volatility_threshold": {
                "type": PineScriptType.FLOAT,
                "title": "Volatility Threshold",
                "group": "Volatility",
                "minval": 0.001,
                "maxval": 0.1,
                "step": 0.001,
                "tooltip": "Volatility threshold for signal filtering"
            },
            
            # Timing parameters
            "entry_delay": {
                "type": PineScriptType.INT,
                "title": "Entry Delay",
                "group": "Timing",
                "minval": 0,
                "maxval": 10,
                "step": 1,
                "tooltip": "Bars to wait before entry"
            },
            "exit_delay": {
                "type": PineScriptType.INT,
                "title": "Exit Delay",
                "group": "Timing",
                "minval": 0,
                "maxval": 10,
                "step": 1,
                "tooltip": "Bars to wait before exit"
            },
            
            # Risk management
            "stop_loss": {
                "type": PineScriptType.FLOAT,
                "title": "Stop Loss %",
                "group": "Risk Management",
                "minval": 0.1,
                "maxval": 10.0,
                "step": 0.1,
                "tooltip": "Stop loss percentage"
            },
            "take_profit": {
                "type": PineScriptType.FLOAT,
                "title": "Take Profit %",
                "group": "Risk Management",
                "minval": 0.1,
                "maxval": 20.0,
                "step": 0.1,
                "tooltip": "Take profit percentage"
            },
            "position_size": {
                "type": PineScriptType.FLOAT,
                "title": "Position Size %",
                "group": "Risk Management",
                "minval": 1.0,
                "maxval": 100.0,
                "step": 1.0,
                "tooltip": "Position size as percentage of equity"
            },
            "max_risk": {
                "type": PineScriptType.FLOAT,
                "title": "Max Risk %",
                "group": "Risk Management",
                "minval": 0.1,
                "maxval": 5.0,
                "step": 0.1,
                "tooltip": "Maximum risk per trade"
            }
        }
        
        # Pattern-based type detection
        self.type_patterns = [
            (r".*period.*", PineScriptType.INT),
            (r".*length.*", PineScriptType.INT),
            (r".*window.*", PineScriptType.INT),
            (r".*threshold.*", PineScriptType.FLOAT),
            (r".*pct.*", PineScriptType.FLOAT),
            (r".*percent.*", PineScriptType.FLOAT),
            (r".*ratio.*", PineScriptType.FLOAT),
            (r".*factor.*", PineScriptType.FLOAT),
            (r".*enable.*", PineScriptType.BOOL),
            (r".*use_.*", PineScriptType.BOOL),
            (r".*allow_.*", PineScriptType.BOOL),
        ]
    
    def translate_parameter(self, name: str, value: Any) -> PineScriptParameter:
        """
        Translate a Python parameter to Pine Script input parameter.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Returns:
            PineScriptParameter object
        """
        # Clean parameter name for Pine Script
        clean_name = self._clean_parameter_name(name)
        
        # Check if we have a predefined rule for this parameter
        if name in self.parameter_rules:
            rule = self.parameter_rules[name]
            return PineScriptParameter(
                name=clean_name,
                param_type=rule["type"],
                default_value=value,
                title=rule["title"],
                tooltip=rule.get("tooltip", ""),
                group=rule.get("group", "Strategy Parameters"),
                minval=rule.get("minval"),
                maxval=rule.get("maxval"),
                step=rule.get("step"),
                options=rule.get("options")
            )
        
        # Auto-detect parameter type and constraints
        param_type = self._detect_parameter_type(name, value)
        title = self._generate_title(name)
        group = self._detect_group(name)
        minval, maxval, step = self._suggest_constraints(name, value, param_type)
        tooltip = self._generate_tooltip(name, param_type)
        
        return PineScriptParameter(
            name=clean_name,
            param_type=param_type,
            default_value=value,
            title=title,
            tooltip=tooltip,
            group=group,
            minval=minval,
            maxval=maxval,
            step=step
        )
    
    def translate_parameters(self, parameters: Dict[str, Any]) -> List[PineScriptParameter]:
        """Translate multiple parameters."""
        pine_params = []
        
        for name, value in parameters.items():
            try:
                pine_param = self.translate_parameter(name, value)
                pine_params.append(pine_param)
            except Exception as e:
                self.logger.warning(f"Failed to translate parameter {name}: {e}")
        
        # Sort parameters by group and name for better organization
        pine_params.sort(key=lambda p: (p.group, p.name))
        
        return pine_params
    
    def _clean_parameter_name(self, name: str) -> str:
        """Clean parameter name for Pine Script variable naming."""
        # Replace invalid characters and ensure valid Pine Script identifier
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
        
        # Ensure it doesn't start with a number
        if clean_name and clean_name[0].isdigit():
            clean_name = f"param_{clean_name}"
        
        # Ensure it's not empty
        if not clean_name:
            clean_name = "param"
        
        return clean_name
    
    def _detect_parameter_type(self, name: str, value: Any) -> PineScriptType:
        """Detect appropriate Pine Script type for a parameter."""
        
        # Type detection based on value type
        if isinstance(value, bool):
            return PineScriptType.BOOL
        elif isinstance(value, int):
            return PineScriptType.INT
        elif isinstance(value, float):
            return PineScriptType.FLOAT
        elif isinstance(value, str):
            return PineScriptType.STRING
        
        # Pattern-based detection for parameter names
        name_lower = name.lower()
        for pattern, param_type in self.type_patterns:
            if re.match(pattern, name_lower):
                return param_type
        
        # Default to float for numeric-like parameters
        return PineScriptType.FLOAT
    
    def _generate_title(self, name: str) -> str:
        """Generate a human-readable title from parameter name."""
        # Split on underscores and capitalize words
        words = name.replace('_', ' ').split()
        
        # Capitalize each word
        title_words = []
        for word in words:
            if word.upper() in ['RSI', 'MACD', 'SMA', 'EMA', 'WMA', 'RMA', 'BB']:
                title_words.append(word.upper())
            else:
                title_words.append(word.capitalize())
        
        return ' '.join(title_words)
    
    def _detect_group(self, name: str) -> str:
        """Detect appropriate parameter group."""
        
        name_lower = name.lower()
        
        # Group detection patterns
        if any(term in name_lower for term in ['rsi', 'relative_strength']):
            return "RSI"
        elif any(term in name_lower for term in ['macd', 'convergence']):
            return "MACD"
        elif any(term in name_lower for term in ['ma', 'moving_average', 'sma', 'ema', 'wma']):
            return "Moving Average"
        elif any(term in name_lower for term in ['bb', 'bollinger']):
            return "Bollinger Bands"
        elif any(term in name_lower for term in ['volume', 'vol']):
            return "Volume"
        elif any(term in name_lower for term in ['volatility', 'atr']):
            return "Volatility"
        elif any(term in name_lower for term in ['stop', 'take_profit', 'risk', 'position']):
            return "Risk Management"
        elif any(term in name_lower for term in ['entry', 'exit', 'signal']):
            return "Signals"
        elif any(term in name_lower for term in ['time', 'delay', 'bars']):
            return "Timing"
        else:
            return "Strategy Parameters"
    
    def _suggest_constraints(
        self, 
        name: str, 
        value: Any, 
        param_type: PineScriptType
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Suggest appropriate constraints for a parameter."""
        
        name_lower = name.lower()
        
        # Constraint suggestions based on parameter name patterns
        if param_type == PineScriptType.INT:
            if 'period' in name_lower or 'length' in name_lower:
                return 1, 500, 1
            elif 'delay' in name_lower:
                return 0, 10, 1
            else:
                return 1, 100, 1
        
        elif param_type == PineScriptType.FLOAT:
            if 'threshold' in name_lower:
                return 0.0, 1.0, 0.01
            elif 'pct' in name_lower or 'percent' in name_lower:
                return 0.1, 100.0, 0.1
            elif 'ratio' in name_lower or 'factor' in name_lower:
                return 0.1, 10.0, 0.1
            elif 'stop' in name_lower:
                return 0.1, 10.0, 0.1
            elif 'profit' in name_lower:
                return 0.1, 20.0, 0.1
            elif 'risk' in name_lower:
                return 0.1, 5.0, 0.1
            else:
                return None, None, 0.01
        
        return None, None, None
    
    def _generate_tooltip(self, name: str, param_type: PineScriptType) -> str:
        """Generate a helpful tooltip for the parameter."""
        
        name_lower = name.lower()
        
        # Tooltip suggestions based on parameter patterns
        if 'period' in name_lower:
            return f"Number of bars for {name.replace('_', ' ')} calculation"
        elif 'threshold' in name_lower:
            return f"Threshold value for {name.replace('_', ' ')}"
        elif 'pct' in name_lower or 'percent' in name_lower:
            return f"Percentage value for {name.replace('_', ' ')}"
        elif 'stop' in name_lower:
            return "Stop loss as percentage of entry price"
        elif 'profit' in name_lower:
            return "Take profit as percentage of entry price"
        elif 'delay' in name_lower:
            return f"Number of bars to delay for {name.replace('_', ' ')}"
        else:
            return f"Parameter for {name.replace('_', ' ')}"
    
    def add_parameter_rule(self, name: str, rule: Dict[str, Any]):
        """Add a custom parameter translation rule."""
        self.parameter_rules[name] = rule
        self.logger.info(f"Added parameter rule for: {name}")
    
    def get_parameter_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get all parameter translation rules."""
        return self.parameter_rules.copy()
    
    def export_parameters_as_pine_script(self, parameters: Dict[str, Any]) -> str:
        """Export parameters as complete Pine Script input section."""
        
        pine_params = self.translate_parameters(parameters)
        
        if not pine_params:
            return "// No parameters to export"
        
        sections = {}
        
        # Group parameters by group
        for param in pine_params:
            group = param.group
            if group not in sections:
                sections[group] = []
            sections[group].append(param.to_pine_script())
        
        # Build the complete input section
        output_lines = ["// === INPUT PARAMETERS ==="]
        
        for group_name, param_lines in sections.items():
            output_lines.append(f"\n// {group_name}")
            output_lines.extend(param_lines)
        
        return "\n".join(output_lines) 
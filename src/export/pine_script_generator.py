"""
Pine Script Generator

This module provides the main PineScriptGenerator class that converts optimized
Python trading strategies into TradingView Pine Script v5 code for live trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import re
import logging

try:
    from ..strategies.base_strategy import BaseStrategy
    from ..optimization.hyperopt_optimizer import OptimizationResult
    from ..utils.logger import get_logger
    from .parameter_translator import ParameterTranslator, PineScriptParameter
except ImportError:
    from src.strategies.base_strategy import BaseStrategy
    from src.optimization.hyperopt_optimizer import OptimizationResult
    from src.utils.logger import get_logger
    from src.export.parameter_translator import ParameterTranslator, PineScriptParameter


@dataclass
class PineScriptConfig:
    """Configuration for Pine Script generation."""
    
    # Pine Script metadata
    version: str = "5"
    title: str = ""
    short_title: str = ""
    overlay: bool = False
    precision: int = 4
    
    # Strategy settings
    initial_capital: float = 100000.0
    default_qty_type: str = "strategy.percent_of_equity"
    default_qty_value: float = 100.0
    commission_type: str = "strategy.commission.percent"
    commission_value: float = 0.1
    slippage: int = 3
    
    # Risk management
    include_stop_loss: bool = True
    include_take_profit: bool = True
    include_position_sizing: bool = True
    include_risk_management: bool = True
    
    # Features
    include_alerts: bool = True
    include_debugging: bool = True
    include_visualization: bool = True
    add_performance_table: bool = True
    
    # Output settings
    output_directory: str = "exports/pine_script"
    file_extension: str = ".pine"


@dataclass 
class SignalCondition:
    """Represents a trading signal condition in Pine Script."""
    
    name: str
    condition: str
    description: str
    signal_type: str  # "long_entry", "long_exit", "short_entry", "short_exit"


@dataclass
class IndicatorCalculation:
    """Represents an indicator calculation in Pine Script."""
    
    name: str
    calculation: str
    description: str
    plot: bool = False
    plot_color: str = "color.blue"


class PineScriptTemplate(ABC):
    """Abstract base class for Pine Script strategy templates."""
    
    def __init__(self, config: PineScriptConfig):
        self.config = config
        self.logger = get_logger(f"pine_template_{self.__class__.__name__}")
    
    @abstractmethod
    def get_template_name(self) -> str:
        """Get the template name."""
        pass
    
    @abstractmethod
    def get_required_parameters(self) -> List[str]:
        """Get list of required parameter names."""
        pass
    
    @abstractmethod
    def generate_indicators(self, params: Dict[str, Any]) -> List[IndicatorCalculation]:
        """Generate indicator calculations for the strategy."""
        pass
    
    @abstractmethod
    def generate_signals(self, params: Dict[str, Any]) -> List[SignalCondition]:
        """Generate trading signal conditions."""
        pass
    
    def get_template_description(self) -> str:
        """Get template description."""
        return f"Pine Script template for {self.get_template_name()}"


class MovingAverageCrossoverTemplate(PineScriptTemplate):
    """Template for Moving Average Crossover strategies."""
    
    def get_template_name(self) -> str:
        return "Moving Average Crossover"
    
    def get_required_parameters(self) -> List[str]:
        return ["fast_period", "slow_period", "signal_threshold"]
    
    def generate_indicators(self, params: Dict[str, Any]) -> List[IndicatorCalculation]:
        fast_period = params.get("fast_period", 12)
        slow_period = params.get("slow_period", 26)
        
        return [
            IndicatorCalculation(
                name="fast_ma",
                calculation=f"ta.sma(close, {fast_period})",
                description=f"Fast Moving Average ({fast_period} periods)",
                plot=True,
                plot_color="color.blue"
            ),
            IndicatorCalculation(
                name="slow_ma", 
                calculation=f"ta.sma(close, {slow_period})",
                description=f"Slow Moving Average ({slow_period} periods)",
                plot=True,
                plot_color="color.red"
            ),
            IndicatorCalculation(
                name="signal_strength",
                calculation="math.abs(fast_ma - slow_ma) / slow_ma",
                description="Signal strength as percentage difference",
                plot=False
            )
        ]
    
    def generate_signals(self, params: Dict[str, Any]) -> List[SignalCondition]:
        threshold = params.get("signal_threshold", 0.02)
        
        return [
            SignalCondition(
                name="long_entry",
                condition=f"ta.crossover(fast_ma, slow_ma) and signal_strength > {threshold}",
                description=f"Fast MA crosses above slow MA with strength > {threshold*100}%",
                signal_type="long_entry"
            ),
            SignalCondition(
                name="long_exit",
                condition="ta.crossunder(fast_ma, slow_ma)",
                description="Fast MA crosses below slow MA",
                signal_type="long_exit"
            ),
            SignalCondition(
                name="short_entry",
                condition=f"ta.crossunder(fast_ma, slow_ma) and signal_strength > {threshold}",
                description=f"Fast MA crosses below slow MA with strength > {threshold*100}%",
                signal_type="short_entry"
            ),
            SignalCondition(
                name="short_exit",
                condition="ta.crossover(fast_ma, slow_ma)",
                description="Fast MA crosses above slow MA",
                signal_type="short_exit"
            )
        ]


class RSITemplate(PineScriptTemplate):
    """Template for RSI-based strategies."""
    
    def get_template_name(self) -> str:
        return "RSI Strategy"
    
    def get_required_parameters(self) -> List[str]:
        return ["rsi_period", "rsi_overbought", "rsi_oversold", "confirmation_period"]
    
    def generate_indicators(self, params: Dict[str, Any]) -> List[IndicatorCalculation]:
        rsi_period = params.get("rsi_period", 14)
        confirmation_period = params.get("confirmation_period", 2)
        
        return [
            IndicatorCalculation(
                name="rsi",
                calculation=f"ta.rsi(close, {rsi_period})",
                description=f"RSI ({rsi_period} periods)",
                plot=True,
                plot_color="color.purple"
            ),
            IndicatorCalculation(
                name="rsi_ma",
                calculation=f"ta.sma(rsi, {confirmation_period})",
                description=f"RSI smoothed ({confirmation_period} periods)",
                plot=True,
                plot_color="color.orange"
            )
        ]
    
    def generate_signals(self, params: Dict[str, Any]) -> List[SignalCondition]:
        overbought = params.get("rsi_overbought", 70)
        oversold = params.get("rsi_oversold", 30)
        
        return [
            SignalCondition(
                name="long_entry",
                condition=f"ta.crossover(rsi, {oversold}) and rsi_ma < {oversold + 5}",
                description=f"RSI crosses above {oversold} (oversold recovery)",
                signal_type="long_entry"
            ),
            SignalCondition(
                name="long_exit",
                condition=f"ta.crossunder(rsi, {overbought})",
                description=f"RSI crosses below {overbought} (profit taking)",
                signal_type="long_exit"
            ),
            SignalCondition(
                name="short_entry",
                condition=f"ta.crossunder(rsi, {overbought}) and rsi_ma > {overbought - 5}",
                description=f"RSI crosses below {overbought} (overbought reversal)",
                signal_type="short_entry"
            ),
            SignalCondition(
                name="short_exit",
                condition=f"ta.crossover(rsi, {oversold})",
                description=f"RSI crosses above {oversold} (cover shorts)",
                signal_type="short_exit"
            )
        ]


class MACDTemplate(PineScriptTemplate):
    """Template for MACD-based strategies."""
    
    def get_template_name(self) -> str:
        return "MACD Strategy"
    
    def get_required_parameters(self) -> List[str]:
        return ["macd_fast", "macd_slow", "macd_signal", "signal_threshold"]
    
    def generate_indicators(self, params: Dict[str, Any]) -> List[IndicatorCalculation]:
        fast = params.get("macd_fast", 12)
        slow = params.get("macd_slow", 26) 
        signal = params.get("macd_signal", 9)
        
        return [
            IndicatorCalculation(
                name="[macd_line, signal_line, histogram]",
                calculation=f"ta.macd(close, {fast}, {slow}, {signal})",
                description=f"MACD ({fast}, {slow}, {signal})",
                plot=False
            ),
            IndicatorCalculation(
                name="macd_momentum",
                calculation="macd_line - signal_line",
                description="MACD momentum (MACD - Signal)",
                plot=True,
                plot_color="color.blue"
            )
        ]
    
    def generate_signals(self, params: Dict[str, Any]) -> List[SignalCondition]:
        threshold = params.get("signal_threshold", 0.001)
        
        return [
            SignalCondition(
                name="long_entry",
                condition=f"ta.crossover(macd_line, signal_line) and macd_line > {threshold}",
                description=f"MACD crosses above signal line with strength > {threshold}",
                signal_type="long_entry"
            ),
            SignalCondition(
                name="long_exit",
                condition="ta.crossunder(macd_line, signal_line)",
                description="MACD crosses below signal line",
                signal_type="long_exit"
            ),
            SignalCondition(
                name="short_entry", 
                condition=f"ta.crossunder(macd_line, signal_line) and macd_line < -{threshold}",
                description=f"MACD crosses below signal line with strength < -{threshold}",
                signal_type="short_entry"
            ),
            SignalCondition(
                name="short_exit",
                condition="ta.crossover(macd_line, signal_line)",
                description="MACD crosses above signal line",
                signal_type="short_exit"
            )
        ]


class PineScriptGenerator:
    """Main Pine Script generator that converts Python strategies to Pine Script v5."""
    
    def __init__(self, config: PineScriptConfig = None):
        """
        Initialize the Pine Script generator.
        
        Args:
            config: Pine Script generation configuration
        """
        self.config = config or PineScriptConfig()
        self.logger = get_logger("pine_script_generator")
        self.parameter_translator = ParameterTranslator()
        
        # Create output directory
        Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize strategy templates
        self.templates = {
            "moving_average_crossover": MovingAverageCrossoverTemplate(self.config),
            "rsi_strategy": RSITemplate(self.config),
            "macd_strategy": MACDTemplate(self.config)
        }
        
        self.logger.info(f"Initialized Pine Script generator with {len(self.templates)} templates")
    
    def generate_strategy_script(
        self,
        strategy: BaseStrategy,
        optimization_result: OptimizationResult = None,
        template_name: str = None,
        output_filename: str = None
    ) -> str:
        """
        Generate a complete Pine Script strategy.
        
        Args:
            strategy: Python trading strategy
            optimization_result: Optional optimization results with best parameters
            template_name: Template to use (auto-detected if None)
            output_filename: Output filename (auto-generated if None)
            
        Returns:
            Path to the generated Pine Script file
        """
        self.logger.info(f"Generating Pine Script for strategy: {strategy.name}")
        
        # Get optimized parameters
        params = self._get_strategy_parameters(strategy, optimization_result)
        
        # Auto-detect template if not specified
        if template_name is None:
            template_name = self._detect_template(strategy)
        
        # Get template
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(self.templates.keys())}")
        
        # Generate script components
        pine_script = self._build_complete_script(strategy, template, params)
        
        # Generate output filename
        if output_filename is None:
            output_filename = f"{strategy.name.lower().replace(' ', '_')}_strategy{self.config.file_extension}"
        
        # Write to file
        output_path = Path(self.config.output_directory) / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pine_script)
        
        self.logger.info(f"Generated Pine Script: {output_path}")
        return str(output_path)
    
    def generate_indicator_script(
        self,
        strategy: BaseStrategy,
        optimization_result: OptimizationResult = None,
        template_name: str = None,
        output_filename: str = None
    ) -> str:
        """Generate an indicator-only Pine Script (no strategy, just signals)."""
        
        # Temporarily modify config for indicator mode
        original_overlay = self.config.overlay
        self.config.overlay = True
        
        try:
            # Generate strategy script but modify it for indicator mode
            script_path = self.generate_strategy_script(
                strategy, optimization_result, template_name, output_filename
            )
            
            # Read and modify the script for indicator mode
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            # Convert strategy to indicator
            indicator_script = self._convert_to_indicator(script_content)
            
            # Write indicator version
            if output_filename is None:
                output_filename = f"{strategy.name.lower().replace(' ', '_')}_indicator{self.config.file_extension}"
            
            indicator_path = Path(self.config.output_directory) / output_filename
            with open(indicator_path, 'w', encoding='utf-8') as f:
                f.write(indicator_script)
            
            self.logger.info(f"Generated Pine Script indicator: {indicator_path}")
            return str(indicator_path)
            
        finally:
            self.config.overlay = original_overlay
    
    def _get_strategy_parameters(
        self,
        strategy: BaseStrategy,
        optimization_result: OptimizationResult = None
    ) -> Dict[str, Any]:
        """Get strategy parameters from optimization result or defaults."""
        
        if optimization_result and optimization_result.best_parameters:
            # Use optimized parameters
            params = optimization_result.best_parameters.copy()
            self.logger.info(f"Using optimized parameters: {len(params)} parameters")
        else:
            # Use default strategy parameters
            params = strategy.parameters.copy()
            self.logger.info(f"Using default parameters: {len(params)} parameters")
        
        # Add risk management parameters if available
        if hasattr(strategy, 'risk_params') and strategy.risk_params:
            params.update(strategy.risk_params)
        
        return params
    
    def _detect_template(self, strategy: BaseStrategy) -> str:
        """Auto-detect the appropriate template based on strategy type."""
        
        strategy_name = strategy.name.lower()
        
        # Template detection based on strategy name
        if any(term in strategy_name for term in ["moving_average", "ma_cross", "sma", "ema"]):
            return "moving_average_crossover"
        elif any(term in strategy_name for term in ["rsi", "relative_strength"]):
            return "rsi_strategy"
        elif any(term in strategy_name for term in ["macd", "moving_average_convergence"]):
            return "macd_strategy"
        else:
            # Default to moving average crossover
            self.logger.warning(f"Could not detect template for {strategy.name}, using moving_average_crossover")
            return "moving_average_crossover"
    
    def _build_complete_script(
        self,
        strategy: BaseStrategy,
        template: PineScriptTemplate,
        params: Dict[str, Any]
    ) -> str:
        """Build the complete Pine Script code."""
        
        # Generate script components
        header = self._generate_header(strategy, template)
        inputs = self._generate_inputs(template, params)
        indicators = self._generate_indicators(template, params)
        signals = self._generate_signals(template, params)
        
        # Get the actual signal objects for strategy logic
        signal_objects = template.generate_signals(params)
        strategy_logic = self._generate_strategy_logic(signal_objects)
        
        risk_management = self._generate_risk_management(params)
        visualization = self._generate_visualization(template, params)
        alerts = self._generate_alerts(signal_objects) if self.config.include_alerts else ""
        debugging = self._generate_debugging(strategy, params) if self.config.include_debugging else ""
        
        # Combine all components
        script_parts = [
            header,
            inputs,
            indicators,
            signals,
            strategy_logic,
            risk_management,
            visualization,
            alerts,
            debugging
        ]
        
        return "\n\n".join(part for part in script_parts if part.strip())
    
    def _generate_header(self, strategy: BaseStrategy, template: PineScriptTemplate) -> str:
        """Generate Pine Script header."""
        
        title = self.config.title or f"{strategy.name} - {template.get_template_name()}"
        short_title = self.config.short_title or strategy.name[:20]
        
        header = f'''// This Pine Script was automatically generated by AI Trading Strategy Optimization System
// Strategy: {strategy.name}
// Template: {template.get_template_name()}
// Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
// 
// DISCLAIMER: This is an automated strategy conversion. Please review and test thoroughly
// before using with real capital. Past performance does not guarantee future results.

//@version={self.config.version}
strategy(
    title="{title}",
    shorttitle="{short_title}",
    overlay={str(self.config.overlay).lower()},
    initial_capital={self.config.initial_capital},
    default_qty_type={self.config.default_qty_type},
    default_qty_value={self.config.default_qty_value},
    commission_type={self.config.commission_type},
    commission_value={self.config.commission_value},
    slippage={self.config.slippage},
    precision={self.config.precision}
)'''
        
        return header
    
    def _generate_inputs(self, template: PineScriptTemplate, params: Dict[str, Any]) -> str:
        """Generate Pine Script input parameters."""
        
        inputs_section = "// === INPUT PARAMETERS ==="
        
        # Translate parameters using the parameter translator
        pine_params = []
        for param_name in template.get_required_parameters():
            if param_name in params:
                pine_param = self.parameter_translator.translate_parameter(
                    param_name, params[param_name]
                )
                pine_params.append(pine_param.to_pine_script())
        
        # Add risk management inputs if enabled
        if self.config.include_risk_management:
            pine_params.extend([
                'stop_loss_pct = input.float(2.0, "Stop Loss %", minval=0.1, maxval=10.0, step=0.1, group="Risk Management")',
                'take_profit_pct = input.float(4.0, "Take Profit %", minval=0.1, maxval=20.0, step=0.1, group="Risk Management")',
                'position_size_pct = input.float(100.0, "Position Size %", minval=1.0, maxval=100.0, step=1.0, group="Risk Management")'
            ])
        
        if pine_params:
            inputs_section += "\n" + "\n".join(pine_params)
        
        return inputs_section
    
    def _generate_indicators(self, template: PineScriptTemplate, params: Dict[str, Any]) -> str:
        """Generate indicator calculations."""
        
        indicators_section = "// === INDICATOR CALCULATIONS ==="
        
        indicators = template.generate_indicators(params)
        
        for indicator in indicators:
            indicators_section += f"\n// {indicator.description}"
            indicators_section += f"\n{indicator.name} = {indicator.calculation}"
        
        return indicators_section
    
    def _generate_signals(self, template: PineScriptTemplate, params: Dict[str, Any]) -> str:
        """Generate trading signal conditions."""
        
        signals_section = "// === TRADING SIGNALS ==="
        
        signals = template.generate_signals(params)
        
        for signal in signals:
            signals_section += f"\n// {signal.description}"
            signals_section += f"\n{signal.name} = {signal.condition}"
        
        return signals_section
    
    def _generate_strategy_logic(self, signals: List[SignalCondition]) -> str:
        """Generate strategy entry/exit logic."""
        
        logic_section = "// === STRATEGY LOGIC ==="
        
        # Find signal conditions by type
        long_entry = next((s.name for s in signals if s.signal_type == "long_entry"), None)
        long_exit = next((s.name for s in signals if s.signal_type == "long_exit"), None)
        short_entry = next((s.name for s in signals if s.signal_type == "short_entry"), None)
        short_exit = next((s.name for s in signals if s.signal_type == "short_exit"), None)
        
        # Generate strategy calls
        if long_entry:
            logic_section += f"\nif ({long_entry})"
            logic_section += f"\n    strategy.entry(\"Long\", strategy.long, qty=position_size_pct/100 * strategy.equity / close)"
        
        if long_exit:
            logic_section += f"\nif ({long_exit})"
            logic_section += f"\n    strategy.close(\"Long\")"
        
        if short_entry:
            logic_section += f"\nif ({short_entry})"
            logic_section += f"\n    strategy.entry(\"Short\", strategy.short, qty=position_size_pct/100 * strategy.equity / close)"
        
        if short_exit:
            logic_section += f"\nif ({short_exit})"
            logic_section += f"\n    strategy.close(\"Short\")"
        
        return logic_section
    
    def _generate_risk_management(self, params: Dict[str, Any]) -> str:
        """Generate risk management code."""
        
        if not self.config.include_risk_management:
            return ""
        
        risk_section = "// === RISK MANAGEMENT ==="
        risk_section += "\n// Stop loss and take profit"
        risk_section += "\nif (strategy.position_size > 0)  // Long position"
        risk_section += "\n    strategy.exit(\"Long SL/TP\", \"Long\", stop=close * (1 - stop_loss_pct/100), limit=close * (1 + take_profit_pct/100))"
        risk_section += "\nif (strategy.position_size < 0)  // Short position"
        risk_section += "\n    strategy.exit(\"Short SL/TP\", \"Short\", stop=close * (1 + stop_loss_pct/100), limit=close * (1 - take_profit_pct/100))"
        
        return risk_section
    
    def _generate_visualization(self, template: PineScriptTemplate, params: Dict[str, Any]) -> str:
        """Generate visualization plots."""
        
        if not self.config.include_visualization:
            return ""
        
        viz_section = "// === VISUALIZATION ==="
        
        # Plot indicators that are marked for plotting
        indicators = template.generate_indicators(params)
        for indicator in indicators:
            if indicator.plot:
                viz_section += f"\nplot({indicator.name}, title=\"{indicator.description}\", color={indicator.plot_color})"
        
        # Add signal markers
        signals = template.generate_signals(params)
        for signal in signals:
            if signal.signal_type == "long_entry":
                viz_section += f"\nplotshape({signal.name}, title=\"Long Entry\", location=location.belowbar, style=shape.triangleup, size=size.small, color=color.green)"
            elif signal.signal_type == "short_entry":
                viz_section += f"\nplotshape({signal.name}, title=\"Short Entry\", location=location.abovebar, style=shape.triangledown, size=size.small, color=color.red)"
        
        return viz_section
    
    def _generate_alerts(self, signals: List[SignalCondition]) -> str:
        """Generate alert conditions."""
        
        alerts_section = "// === ALERT CONDITIONS ==="
        
        for signal in signals:
            if signal.signal_type in ["long_entry", "short_entry"]:
                action = "BUY" if signal.signal_type == "long_entry" else "SELL"
                alerts_section += f"\nalertcondition({signal.name}, title=\"{action} Signal\", message=\"{action}: {signal.description}\")"
        
        return alerts_section
    
    def _generate_debugging(self, strategy: BaseStrategy, params: Dict[str, Any]) -> str:
        """Generate debugging and performance information."""
        
        debug_section = "// === DEBUGGING & PERFORMANCE ==="
        
        if self.config.add_performance_table:
            debug_section += "\n// Performance table"
            debug_section += "\nif barstate.islast"
            debug_section += "\n    var table perfTable = table.new(position.top_right, 2, 8, bgcolor=color.white, border_width=1)"
            debug_section += "\n    table.cell(perfTable, 0, 0, \"Metric\", text_color=color.black, bgcolor=color.gray)"
            debug_section += "\n    table.cell(perfTable, 1, 0, \"Value\", text_color=color.black, bgcolor=color.gray)"
            debug_section += "\n    table.cell(perfTable, 0, 1, \"Total Trades\", text_color=color.black)"
            debug_section += "\n    table.cell(perfTable, 1, 1, str.tostring(strategy.closedtrades), text_color=color.black)"
            debug_section += "\n    table.cell(perfTable, 0, 2, \"Win Rate\", text_color=color.black)"
            debug_section += "\n    table.cell(perfTable, 1, 2, str.tostring(strategy.wintrades / strategy.closedtrades * 100, \"#.##\") + \"%\", text_color=color.black)"
            debug_section += "\n    table.cell(perfTable, 0, 3, \"Profit Factor\", text_color=color.black)"
            debug_section += "\n    table.cell(perfTable, 1, 3, str.tostring(strategy.grossprofit / strategy.grossloss, \"#.##\"), text_color=color.black)"
            debug_section += "\n    table.cell(perfTable, 0, 4, \"Net Profit\", text_color=color.black)"
            debug_section += "\n    table.cell(perfTable, 1, 4, str.tostring(strategy.netprofit, \"#.##\"), text_color=color.black)"
        
        # Add parameter display
        debug_section += "\n// Parameter validation"
        debug_section += "\nif barstate.islast and strategy.closedtrades == 0"
        debug_section += "\n    runtime.error(\"No trades generated. Check parameters and market conditions.\")"
        
        return debug_section
    
    def _convert_to_indicator(self, strategy_script: str) -> str:
        """Convert a strategy script to an indicator script."""
        
        # Replace strategy() with indicator()
        indicator_script = re.sub(
            r'strategy\s*\(',
            'indicator(',
            strategy_script
        )
        
        # Remove strategy-specific parameters
        strategy_params_to_remove = [
            r'initial_capital=\d+\.?\d*,?\s*',
            r'default_qty_type=[^,]+,?\s*',
            r'default_qty_value=\d+\.?\d*,?\s*',
            r'commission_type=[^,]+,?\s*',
            r'commission_value=\d+\.?\d*,?\s*',
            r'slippage=\d+,?\s*'
        ]
        
        for param_pattern in strategy_params_to_remove:
            indicator_script = re.sub(param_pattern, '', indicator_script)
        
        # Remove strategy logic sections
        sections_to_remove = [
            r'// === STRATEGY LOGIC ===.*?(?=// ===|\n$|\Z)',
            r'// === RISK MANAGEMENT ===.*?(?=// ===|\n$|\Z)',
            r'strategy\.entry\([^)]+\)',
            r'strategy\.close\([^)]+\)',
            r'strategy\.exit\([^)]+\)'
        ]
        
        for section_pattern in sections_to_remove:
            indicator_script = re.sub(section_pattern, '', indicator_script, flags=re.DOTALL)
        
        # Clean up extra whitespace
        indicator_script = re.sub(r'\n{3,}', '\n\n', indicator_script)
        
        return indicator_script
    
    def get_available_templates(self) -> List[str]:
        """Get list of available strategy templates."""
        return list(self.templates.keys())
    
    def add_custom_template(self, name: str, template: PineScriptTemplate):
        """Add a custom strategy template."""
        self.templates[name] = template
        self.logger.info(f"Added custom template: {name}")
    
    def generate_batch_scripts(
        self,
        strategies: List[Tuple[BaseStrategy, Optional[OptimizationResult]]],
        output_directory: str = None
    ) -> List[str]:
        """Generate Pine Scripts for multiple strategies."""
        
        if output_directory:
            original_dir = self.config.output_directory
            self.config.output_directory = output_directory
        
        generated_files = []
        
        try:
            for strategy, optimization_result in strategies:
                try:
                    script_path = self.generate_strategy_script(strategy, optimization_result)
                    generated_files.append(script_path)
                    self.logger.info(f"Generated script for {strategy.name}")
                except Exception as e:
                    self.logger.error(f"Failed to generate script for {strategy.name}: {e}")
            
            self.logger.info(f"Batch generation completed. Generated {len(generated_files)} scripts.")
            return generated_files
            
        finally:
            if output_directory:
                self.config.output_directory = original_dir 
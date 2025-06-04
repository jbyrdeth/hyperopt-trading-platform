"""
Pine Script Validator

This module provides validation functionality for generated Pine Script code,
checking for syntax errors, best practices, and TradingView compliance.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    from ..utils.logger import get_logger
except ImportError:
    from src.utils.logger import get_logger


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue in Pine Script code."""
    
    severity: ValidationSeverity
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    rule: str = ""
    suggestion: str = ""
    
    def __str__(self) -> str:
        location = ""
        if self.line_number is not None:
            location = f" (line {self.line_number}"
            if self.column is not None:
                location += f", col {self.column}"
            location += ")"
        
        return f"{self.severity.value.upper()}: {self.message}{location}"


@dataclass
class ValidationResult:
    """Complete validation result for Pine Script code."""
    
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue and update counters."""
        self.issues.append(issue)
        
        if issue.severity == ValidationSeverity.ERROR or issue.severity == ValidationSeverity.CRITICAL:
            self.error_count += 1
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warning_count += 1
        else:
            self.info_count += 1
    
    def get_summary(self) -> str:
        """Get a summary of validation results."""
        if self.is_valid:
            summary = "✅ Pine Script validation passed"
        else:
            summary = "❌ Pine Script validation failed"
        
        summary += f" ({self.error_count} errors, {self.warning_count} warnings, {self.info_count} info)"
        return summary


class PineScriptValidator:
    """Validates Pine Script code for syntax and best practices."""
    
    def __init__(self):
        self.logger = get_logger("pine_validator")
        
        # Pine Script v5 keywords and built-in functions
        self.pine_keywords = {
            'strategy', 'indicator', 'library', 'script',
            'if', 'else', 'for', 'while', 'switch', 'var', 'varip',
            'import', 'export', 'method', 'type', 'true', 'false', 'na',
            'and', 'or', 'not', 'math', 'array', 'matrix', 'map', 'string',
            'color', 'line', 'label', 'table', 'box', 'polyline'
        }
        
        # Required sections for a complete strategy
        self.required_sections = [
            'version',
            'strategy',
            'inputs',
            'calculations',
            'logic'
        ]
        
        # Pine Script v5 built-in variables
        self.builtin_variables = {
            'open', 'high', 'low', 'close', 'volume', 'time', 'timeframe',
            'syminfo', 'session', 'bar_index', 'barstate', 'dayofweek',
            'dayofmonth', 'dayofyear', 'hour', 'minute', 'month', 'second',
            'year', 'weekofyear'
        }
        
        # Strategy-specific functions
        self.strategy_functions = {
            'strategy.entry', 'strategy.exit', 'strategy.close',
            'strategy.cancel', 'strategy.cancel_all', 'strategy.risk'
        }
        
        # Validation rules
        self.validation_rules = [
            self._validate_version,
            self._validate_header,
            self._validate_syntax,
            self._validate_variables,
            self._validate_strategy_calls,
            self._validate_risk_management,
            self._validate_performance,
            self._validate_best_practices
        ]
    
    def validate(self, pine_script: str) -> ValidationResult:
        """
        Validate Pine Script code.
        
        Args:
            pine_script: Pine Script code to validate
            
        Returns:
            ValidationResult with issues and status
        """
        result = ValidationResult(is_valid=True)
        
        if not pine_script or not pine_script.strip():
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message="Pine Script code is empty",
                rule="empty_script"
            ))
            return result
        
        # Split into lines for line-based validation
        lines = pine_script.splitlines()
        
        # Run all validation rules
        for rule_func in self.validation_rules:
            try:
                rule_func(pine_script, lines, result)
            except Exception as e:
                self.logger.warning(f"Validation rule {rule_func.__name__} failed: {e}")
        
        self.logger.info(f"Validation completed: {result.get_summary()}")
        return result
    
    def _validate_version(self, script: str, lines: List[str], result: ValidationResult):
        """Validate Pine Script version declaration."""
        
        version_pattern = r'//@version\s*=\s*(\d+)'
        version_match = re.search(version_pattern, script)
        
        if not version_match:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Missing //@version declaration",
                rule="version_required",
                suggestion="Add '//@version=5' at the top of the script"
            ))
            return
        
        version = int(version_match.group(1))
        if version < 5:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Using Pine Script v{version}, consider upgrading to v5",
                rule="version_outdated",
                suggestion="Update to '//@version=5' for latest features"
            ))
        elif version > 5:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Using Pine Script v{version}, which may not be supported",
                rule="version_unsupported"
            ))
    
    def _validate_header(self, script: str, lines: List[str], result: ValidationResult):
        """Validate strategy/indicator header."""
        
        # Check for strategy() or indicator() declaration
        header_pattern = r'(strategy|indicator)\s*\('
        header_match = re.search(header_pattern, script)
        
        if not header_match:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Missing strategy() or indicator() declaration",
                rule="header_required",
                suggestion="Add strategy() or indicator() function call"
            ))
            return
        
        header_type = header_match.group(1)
        
        # Extract header parameters
        header_start = header_match.start()
        paren_count = 0
        header_end = header_start
        
        for i, char in enumerate(script[header_start:], header_start):
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count == 0:
                    header_end = i + 1
                    break
        
        header_content = script[header_start:header_end]
        
        # Validate required parameters
        required_params = ['title']
        for param in required_params:
            if param not in header_content:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Missing recommended parameter '{param}' in {header_type}()",
                    rule="header_params",
                    suggestion=f"Add {param} parameter for better script identification"
                ))
    
    def _validate_syntax(self, script: str, lines: List[str], result: ValidationResult):
        """Validate basic Pine Script syntax."""
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            if not line or line.startswith('//'):
                continue
            
            # Check for common syntax errors
            
            # Unmatched parentheses
            if line.count('(') != line.count(')'):
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Unmatched parentheses",
                    line_number=line_num,
                    rule="syntax_parentheses"
                ))
            
            # Unmatched brackets
            if line.count('[') != line.count(']'):
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Unmatched brackets",
                    line_number=line_num,
                    rule="syntax_brackets"
                ))
            
            # Invalid variable names
            var_pattern = r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*='
            var_match = re.match(var_pattern, line)
            if var_match:
                var_name = var_match.group(1)
                if var_name in self.pine_keywords:
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Variable name '{var_name}' is a reserved keyword",
                        line_number=line_num,
                        rule="syntax_reserved_word",
                        suggestion=f"Use a different variable name like '{var_name}_val'"
                    ))
    
    def _validate_variables(self, script: str, lines: List[str], result: ValidationResult):
        """Validate variable declarations and usage."""
        
        declared_vars = set()
        used_vars = set()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            if not line or line.startswith('//'):
                continue
            
            # Find variable declarations
            var_decl_pattern = r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*='
            var_match = re.match(var_decl_pattern, line)
            if var_match:
                var_name = var_match.group(1)
                if var_name not in self.builtin_variables:
                    declared_vars.add(var_name)
            
            # Find variable usage
            var_usage_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
            for match in re.finditer(var_usage_pattern, line):
                var_name = match.group(1)
                if (var_name not in self.pine_keywords and 
                    var_name not in self.builtin_variables and
                    '=' not in line[:match.start()]):  # Not a declaration
                    used_vars.add(var_name)
        
        # Check for unused variables
        unused_vars = declared_vars - used_vars
        for var_name in unused_vars:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Variable '{var_name}' is declared but never used",
                rule="variable_unused",
                suggestion=f"Remove unused variable '{var_name}' or use it in calculations"
            ))
        
        # Check for undeclared variables (excluding built-ins)
        undeclared_vars = used_vars - declared_vars - self.builtin_variables
        for var_name in undeclared_vars:
            # Skip function names and common patterns
            if not any(pattern in var_name for pattern in ['ta.', 'math.', 'strategy.', 'input.']):
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Variable '{var_name}' may be used before declaration",
                    rule="variable_undeclared"
                ))
    
    def _validate_strategy_calls(self, script: str, lines: List[str], result: ValidationResult):
        """Validate strategy entry/exit calls."""
        
        if 'strategy(' not in script:
            return  # Skip for indicators
        
        has_entry = False
        has_exit = False
        
        # Check for strategy calls
        strategy_entry_pattern = r'strategy\.entry\s*\('
        strategy_exit_pattern = r'strategy\.(exit|close)\s*\('
        
        if re.search(strategy_entry_pattern, script):
            has_entry = True
        
        if re.search(strategy_exit_pattern, script):
            has_exit = True
        
        if not has_entry:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="No strategy.entry() calls found",
                rule="strategy_entry_missing",
                suggestion="Add strategy.entry() calls to generate trades"
            ))
        
        if not has_exit:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="No explicit exit calls found",
                rule="strategy_exit_missing",
                suggestion="Consider adding strategy.exit() or strategy.close() for better control"
            ))
    
    def _validate_risk_management(self, script: str, lines: List[str], result: ValidationResult):
        """Validate risk management implementation."""
        
        if 'strategy(' not in script:
            return  # Skip for indicators
        
        has_stop_loss = False
        has_take_profit = False
        has_position_sizing = False
        
        # Check for risk management features
        if re.search(r'stop\s*=', script) or 'stop_loss' in script:
            has_stop_loss = True
        
        if re.search(r'limit\s*=', script) or 'take_profit' in script:
            has_take_profit = True
        
        if re.search(r'qty\s*=', script) or 'position_size' in script:
            has_position_sizing = True
        
        if not has_stop_loss:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="No stop loss implementation found",
                rule="risk_stop_loss",
                suggestion="Consider adding stop loss for risk management"
            ))
        
        if not has_take_profit:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="No take profit implementation found",
                rule="risk_take_profit",
                suggestion="Consider adding take profit targets"
            ))
        
        if not has_position_sizing:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="No dynamic position sizing found",
                rule="risk_position_sizing",
                suggestion="Consider implementing dynamic position sizing"
            ))
    
    def _validate_performance(self, script: str, lines: List[str], result: ValidationResult):
        """Validate performance-related aspects."""
        
        # Check for excessive calculations in loops
        loop_pattern = r'for\s+\w+\s*=\s*\d+\s+to\s+(\d+)'
        for match in re.finditer(loop_pattern, script):
            loop_size = int(match.group(1))
            if loop_size > 500:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Large loop detected (size: {loop_size})",
                    rule="performance_loop_size",
                    suggestion="Consider optimizing large loops for better performance"
                ))
        
        # Check for potential security issues
        if 'request.security' in script:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Using request.security() - ensure proper lookahead handling",
                rule="performance_security",
                suggestion="Review lookahead settings in request.security()"
            ))
    
    def _validate_best_practices(self, script: str, lines: List[str], result: ValidationResult):
        """Validate Pine Script best practices."""
        
        # Check for meaningful variable names
        short_var_pattern = r'^\s*([a-z]{1,2})\s*='
        for line_num, line in enumerate(lines, 1):
            match = re.match(short_var_pattern, line.strip())
            if match and match.group(1) not in ['x', 'y', 'i', 'j']:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Consider using more descriptive variable name than '{match.group(1)}'",
                    line_number=line_num,
                    rule="best_practice_naming"
                ))
        
        # Check for comments
        comment_lines = sum(1 for line in lines if line.strip().startswith('//'))
        total_lines = len([line for line in lines if line.strip()])
        
        if total_lines > 50 and comment_lines / total_lines < 0.1:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Consider adding more comments for better code documentation",
                rule="best_practice_comments",
                suggestion="Add comments to explain complex logic"
            ))
        
        # Check for magic numbers
        magic_number_pattern = r'\b\d{3,}\b'
        for line_num, line in enumerate(lines, 1):
            if re.search(magic_number_pattern, line) and not line.strip().startswith('//'):
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="Consider using named constants instead of magic numbers",
                    line_number=line_num,
                    rule="best_practice_magic_numbers",
                    suggestion="Define constants for large numeric values"
                ))
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """Validate Pine Script file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.validate(content)
        except Exception as e:
            result = ValidationResult(is_valid=False)
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message=f"Failed to read file: {e}",
                rule="file_error"
            ))
            return result
    
    def get_validation_summary(self, result: ValidationResult) -> str:
        """Generate a detailed validation summary."""
        summary = [result.get_summary()]
        
        if result.issues:
            summary.append("\nIssues found:")
            
            # Group issues by severity
            by_severity = {}
            for issue in result.issues:
                severity = issue.severity.value
                if severity not in by_severity:
                    by_severity[severity] = []
                by_severity[severity].append(issue)
            
            # Display issues by severity
            for severity in ['critical', 'error', 'warning', 'info']:
                if severity in by_severity:
                    summary.append(f"\n{severity.upper()}:")
                    for issue in by_severity[severity]:
                        summary.append(f"  • {issue}")
                        if issue.suggestion:
                            summary.append(f"    💡 {issue.suggestion}")
        
        return "\n".join(summary) 
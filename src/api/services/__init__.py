"""
Services Module

Contains business logic services for the trading strategy optimization API.
"""

from .optimization_service import OptimizationService

# Create singleton instance
optimization_service = OptimizationService()

__all__ = ["OptimizationService", "optimization_service"] 
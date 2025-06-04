"""
API Module

This module provides REST API endpoints for the trading strategy optimization system.
It exposes all optimization, validation, and export capabilities via a FastAPI server.
"""

from .main import app
from .models import *
from .auth import *

__all__ = ['app'] 
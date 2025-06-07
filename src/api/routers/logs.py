"""
Log Monitoring and Analysis Router

Provides endpoints for log aggregation, analysis, and monitoring.
These endpoints support the log aggregation system and provide insights
into application behavior, performance, and errors.
"""

import logging
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import os
from pathlib import Path

from ..monitoring.logging import get_log_aggregator, LogLevel, LogCategory, LogContext, log_with_context

logger = logging.getLogger("api.logs")

router = APIRouter()


@router.get("/stats")
async def get_log_stats():
    """
    Get log aggregation statistics.
    
    Returns current log statistics including counts by level, category,
    component, and performance metrics.
    """
    try:
        log_aggregator = get_log_aggregator()
        stats = log_aggregator.get_stats()
        
        return {
            "message": "Log statistics retrieved successfully",
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get log stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve log statistics: {str(e)}"
        )


@router.get("/recent")
async def get_recent_logs(
    limit: int = Query(100, ge=1, le=1000, description="Number of logs to retrieve"),
    level: Optional[str] = Query(None, description="Filter by log level"),
    category: Optional[str] = Query(None, description="Filter by log category"),
    component: Optional[str] = Query(None, description="Filter by component")
):
    """
    Get recent log entries with optional filtering.
    
    Supports filtering by log level, category, and component.
    Returns structured log data for analysis and debugging.
    """
    try:
        log_aggregator = get_log_aggregator()
        
        # Validate parameters
        if level and level.upper() not in [l.value for l in LogLevel]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid log level. Must be one of: {[l.value for l in LogLevel]}"
            )
        
        if category and category.lower() not in [c.value for c in LogCategory]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid log category. Must be one of: {[c.value for c in LogCategory]}"
            )
        
        # Get filtered logs
        logs = log_aggregator.get_recent_logs(
            limit=limit,
            level=level.upper() if level else None,
            category=category.lower() if category else None
        )
        
        # Additional component filtering (not implemented in LogAggregator.get_recent_logs)
        if component:
            logs = [log for log in logs if log.get("component") == component]
        
        return {
            "message": f"Retrieved {len(logs)} recent log entries",
            "logs": logs,
            "filters": {
                "limit": limit,
                "level": level,
                "category": category,
                "component": component
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recent logs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve recent logs: {str(e)}"
        )


@router.get("/analysis")
async def get_log_analysis(
    time_window_hours: int = Query(24, ge=1, le=168, description="Analysis time window in hours")
):
    """
    Get log pattern analysis and insights.
    
    Analyzes log patterns over a specified time window to identify:
    - Error patterns and anomalies
    - Performance issues
    - Security concerns
    - Operational recommendations
    """
    try:
        log_aggregator = get_log_aggregator()
        analysis = log_aggregator.analyze_patterns(time_window_hours)
        
        return {
            "message": "Log analysis completed successfully",
            "analysis": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze logs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze log patterns: {str(e)}"
        )


@router.post("/flush")
async def flush_logs():
    """
    Force flush of log buffer to storage.
    
    Manually triggers the flushing of buffered logs to persistent storage.
    Useful for ensuring logs are persisted before system maintenance.
    """
    try:
        log_aggregator = get_log_aggregator()
        
        # Get buffer size before flush
        buffer_size = len(log_aggregator.logs_buffer)
        
        # Flush logs
        log_aggregator.flush_logs()
        
        return {
            "message": "Log buffer flushed successfully",
            "logs_flushed": buffer_size,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to flush logs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to flush log buffer: {str(e)}"
        )


@router.get("/search")
async def search_logs(
    query: str = Query(..., description="Search query string"),
    limit: int = Query(100, ge=1, le=1000, description="Number of results to return"),
    level: Optional[str] = Query(None, description="Filter by log level"),
    category: Optional[str] = Query(None, description="Filter by log category"),
    since: Optional[str] = Query(None, description="ISO timestamp to search from")
):
    """
    Search logs using text-based queries.
    
    Searches through log messages and context for the specified query string.
    Supports additional filtering by level, category, and time range.
    """
    try:
        log_aggregator = get_log_aggregator()
        
        # Get recent logs for searching (in production, this would query a proper log store)
        logs = log_aggregator.get_recent_logs(limit=limit * 2)  # Get more for filtering
        
        # Parse since timestamp if provided
        since_dt = None
        if since:
            try:
                since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid timestamp format. Use ISO format (e.g., 2023-01-01T00:00:00Z)"
                )
        
        # Filter logs
        filtered_logs = []
        query_lower = query.lower()
        
        for log in logs:
            # Check timestamp filter
            if since_dt:
                log_timestamp = datetime.fromisoformat(log.get("timestamp", "").replace('Z', '+00:00'))
                if log_timestamp < since_dt:
                    continue
            
            # Check level filter
            if level and log.get("level") != level.upper():
                continue
            
            # Check category filter
            if category and log.get("category") != category.lower():
                continue
            
            # Check query match
            message = log.get("message", "").lower()
            context_str = str(log.get("context", {})).lower()
            
            if query_lower in message or query_lower in context_str:
                filtered_logs.append(log)
            
            # Limit results
            if len(filtered_logs) >= limit:
                break
        
        return {
            "message": f"Found {len(filtered_logs)} logs matching query",
            "query": query,
            "logs": filtered_logs,
            "filters": {
                "limit": limit,
                "level": level,
                "category": category,
                "since": since
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search logs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search logs: {str(e)}"
        )


@router.get("/export")
async def export_logs(
    format: str = Query("json", description="Export format (json, csv)"),
    level: Optional[str] = Query(None, description="Filter by log level"),
    category: Optional[str] = Query(None, description="Filter by log category"),
    since: Optional[str] = Query(None, description="ISO timestamp to export from"),
    until: Optional[str] = Query(None, description="ISO timestamp to export until")
):
    """
    Export logs in various formats.
    
    Exports log data in JSON or CSV format with optional filtering.
    Useful for external analysis, compliance, or backup purposes.
    """
    try:
        log_aggregator = get_log_aggregator()
        
        # Validate format
        if format not in ["json", "csv"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid format. Must be 'json' or 'csv'"
            )
        
        # Get logs (in production, this would query the log store with proper time ranges)
        logs = log_aggregator.get_recent_logs(limit=10000)  # Large limit for export
        
        # Apply filters
        filtered_logs = []
        
        # Parse timestamps
        since_dt = None
        until_dt = None
        
        if since:
            try:
                since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid 'since' timestamp format")
        
        if until:
            try:
                until_dt = datetime.fromisoformat(until.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid 'until' timestamp format")
        
        for log in logs:
            # Check timestamp filters
            if since_dt or until_dt:
                log_timestamp = datetime.fromisoformat(log.get("timestamp", "").replace('Z', '+00:00'))
                if since_dt and log_timestamp < since_dt:
                    continue
                if until_dt and log_timestamp > until_dt:
                    continue
            
            # Check level filter
            if level and log.get("level") != level.upper():
                continue
            
            # Check category filter
            if category and log.get("category") != category.lower():
                continue
            
            filtered_logs.append(log)
        
        # Format export data
        if format == "json":
            export_data = {
                "export_info": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "format": format,
                    "total_logs": len(filtered_logs),
                    "filters": {
                        "level": level,
                        "category": category,
                        "since": since,
                        "until": until
                    }
                },
                "logs": filtered_logs
            }
        else:  # CSV format
            import csv
            import io
            
            output = io.StringIO()
            if filtered_logs:
                # Get all possible field names
                fieldnames = set()
                for log in filtered_logs:
                    fieldnames.update(log.keys())
                    if isinstance(log.get("context"), dict):
                        fieldnames.update(f"context.{k}" for k in log["context"].keys())
                
                fieldnames = sorted(list(fieldnames))
                
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                
                for log in filtered_logs:
                    row = log.copy()
                    # Flatten context
                    if isinstance(row.get("context"), dict):
                        for k, v in row["context"].items():
                            row[f"context.{k}"] = v
                        del row["context"]
                    
                    # Convert complex objects to strings
                    for k, v in row.items():
                        if isinstance(v, (dict, list)):
                            row[k] = str(v)
                    
                    writer.writerow(row)
            
            export_data = output.getvalue()
        
        return {
            "message": f"Exported {len(filtered_logs)} logs in {format} format",
            "format": format,
            "total_logs": len(filtered_logs),
            "data": export_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export logs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export logs: {str(e)}"
        )


@router.post("/test")
async def generate_test_logs():
    """
    Generate test log entries for testing and demonstration.
    
    Creates sample log entries with various levels, categories, and patterns
    to test log aggregation, analysis, and alerting systems.
    """
    try:
        log_aggregator = get_log_aggregator()
        
        # Generate test logs with different patterns
        test_logs = [
            {
                "level": "INFO",
                "category": "api",
                "component": "test",
                "message": "Test API request processed successfully",
                "context": {
                    "request_id": "test-001",
                    "duration_ms": 150,
                    "status_code": 200
                }
            },
            {
                "level": "WARNING",
                "category": "performance",
                "component": "test",
                "message": "Slow optimization request detected",
                "context": {
                    "request_id": "test-002",
                    "duration_ms": 6000,
                    "strategy_name": "test_strategy"
                }
            },
            {
                "level": "ERROR",
                "category": "optimization",
                "component": "test",
                "message": "Optimization job failed due to timeout",
                "context": {
                    "job_id": "test-job-003",
                    "error_type": "TimeoutError",
                    "strategy_name": "test_strategy"
                }
            },
            {
                "level": "CRITICAL",
                "category": "security",
                "component": "test",
                "message": "Authentication failed for suspicious request",
                "context": {
                    "request_id": "test-004",
                    "ip_address": "192.168.1.100",
                    "user_agent": "suspicious-bot/1.0"
                }
            },
            {
                "level": "INFO",
                "category": "business",
                "component": "test",
                "message": "Strategy performance analysis completed",
                "context": {
                    "strategy_name": "sma_crossover",
                    "sharpe_ratio": 1.45,
                    "total_return": 15.2
                }
            }
        ]
        
        # Add test logs to aggregator
        for log_data in test_logs:
            # Add timestamp
            log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"
            log_aggregator.add_log(log_data)
        
        # Also log through the standard logger
        test_context = LogContext(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level="INFO",
            category="system",
            component="log_test",
            trace_id="test-trace-001"
        )
        
        log_with_context(
            logger,
            "info",
            f"Generated {len(test_logs)} test log entries for testing",
            test_context,
            test_logs_count=len(test_logs)
        )
        
        return {
            "message": f"Generated {len(test_logs)} test log entries successfully",
            "test_logs": test_logs,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to generate test logs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate test logs: {str(e)}"
        )


@router.get("/files")
async def list_log_files():
    """
    List available log files on the system.
    
    Returns information about log files including size, modification time,
    and location. Useful for log file management and monitoring.
    """
    try:
        log_dir = Path("logs")
        
        if not log_dir.exists():
            return {
                "message": "Log directory does not exist",
                "files": [],
                "timestamp": datetime.utcnow().isoformat()
            }
        
        log_files = []
        
        for file_path in log_dir.iterdir():
            if file_path.is_file() and file_path.suffix in ['.log', '.jsonl']:
                stat = file_path.stat()
                
                log_files.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "is_aggregated": "aggregated" in file_path.name
                })
        
        # Sort by modification time (newest first)
        log_files.sort(key=lambda x: x["modified_time"], reverse=True)
        
        return {
            "message": f"Found {len(log_files)} log files",
            "files": log_files,
            "log_directory": str(log_dir),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to list log files: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list log files: {str(e)}"
        )


@router.get("/config")
async def get_log_config():
    """
    Get current logging configuration.
    
    Returns information about the current logging setup including
    log levels, handlers, formatters, and aggregation settings.
    """
    try:
        # Get current logging configuration
        root_logger = logging.getLogger()
        
        config_info = {
            "root_level": logging.getLevelName(root_logger.level),
            "handlers": [],
            "loggers": {},
            "aggregation": {
                "enabled": True,
                "buffer_size": get_log_aggregator().max_buffer_size,
                "flush_interval": get_log_aggregator().flush_interval
            }
        }
        
        # Get handler information
        for handler in root_logger.handlers:
            handler_info = {
                "type": type(handler).__name__,
                "level": logging.getLevelName(handler.level),
                "formatter": type(handler.formatter).__name__ if handler.formatter else None
            }
            
            # Add file handler specific info
            if hasattr(handler, 'baseFilename'):
                handler_info["filename"] = handler.baseFilename
                if hasattr(handler, 'maxBytes'):
                    handler_info["max_bytes"] = handler.maxBytes
                    handler_info["backup_count"] = handler.backupCount
            
            config_info["handlers"].append(handler_info)
        
        # Get specific logger configurations
        for logger_name in ["api", "api.requests", "uvicorn"]:
            logger_obj = logging.getLogger(logger_name)
            config_info["loggers"][logger_name] = {
                "level": logging.getLevelName(logger_obj.level),
                "handlers_count": len(logger_obj.handlers),
                "propagate": logger_obj.propagate
            }
        
        return {
            "message": "Logging configuration retrieved successfully",
            "config": config_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get log config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve logging configuration: {str(e)}"
        ) 
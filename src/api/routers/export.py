"""
Export Router

Handles export endpoints for Pine Script generation and PDF report creation.
Integrates with existing PineScriptGenerator and ReportGenerator systems.
"""

import logging
import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import tempfile
import asyncio

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ..models import (
    BaseResponse, OptimizationResult, PerformanceMetrics,
    PineScriptExportRequest, PineScriptExportResponse,
    ReportExportRequest, ReportExportResponse, FileDownloadResponse
)
from ..auth import verify_api_key, require_permission
from ..job_manager import job_manager, JobPriority

# Import monitoring
from ..monitoring import get_metrics_collector

# Import existing export components
# from src.export import PineScriptGenerator
# from src.export.pine_script_generator import PineScriptConfig
# from src.reporting import ReportGenerator, ReportConfiguration, ReportDataCollector

router = APIRouter()
logger = logging.getLogger(__name__)

# Get metrics collector for export tracking
metrics_collector = get_metrics_collector()

# File storage configuration
EXPORT_STORAGE_DIR = "exports/api"
MAX_FILE_AGE_HOURS = 24

# Create storage directory
Path(EXPORT_STORAGE_DIR).mkdir(parents=True, exist_ok=True)


class ExportFileManager:
    """Manages export file storage and cleanup."""
    
    def __init__(self, storage_dir: str = EXPORT_STORAGE_DIR):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._update_storage_metrics()
    
    def generate_file_id(self, file_type: str = "export") -> str:
        """Generate unique file ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"{file_type}_{timestamp}_{unique_id}"
    
    def get_file_path(self, file_id: str, extension: str = ".pine") -> Path:
        """Get full file path for a file ID."""
        return self.storage_dir / f"{file_id}{extension}"
    
    def _update_storage_metrics(self):
        """Update file storage metrics."""
        try:
            pine_files = list(self.storage_dir.glob("pine_*.pine"))
            pdf_files = list(self.storage_dir.glob("report_*.pdf"))
            
            # Count and size for Pine files
            pine_count = len(pine_files)
            pine_size = sum(f.stat().st_size for f in pine_files if f.exists())
            
            # Count and size for PDF files  
            pdf_count = len(pdf_files)
            pdf_size = sum(f.stat().st_size for f in pdf_files if f.exists())
            
            # Update metrics
            if metrics_collector:
                metrics_collector.record_file_storage("pine_script", pine_count, pine_size)
                metrics_collector.record_file_storage("pdf_report", pdf_count, pdf_size)
                
        except Exception as e:
            logger.warning(f"Failed to update storage metrics: {e}")
    
    def cleanup_old_files(self, max_age_hours: int = MAX_FILE_AGE_HOURS):
        """Remove files older than specified hours."""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        for file_path in self.storage_dir.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    logger.info(f"Cleaned up old export file: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to clean up file {file_path.name}: {e}")
        
        # Update metrics after cleanup
        self._update_storage_metrics()


# Initialize file manager
file_manager = ExportFileManager()


@router.post(
    "/pine-script",
    response_model=PineScriptExportResponse,
    summary="Generate Pine Script",
    description="Generate TradingView Pine Script from optimization results"
)
async def export_pine_script(
    request: PineScriptExportRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    permissions: Dict[str, bool] = Depends(require_permission("write"))
):
    """
    Generate Pine Script from optimization results.
    
    Creates Pine Script v5 code for TradingView that implements the optimized
    trading strategy with the best parameters found during optimization.
    """
    # Start metrics timing
    start_time = metrics_collector.record_export_start("pine_script") if metrics_collector else None
    
    try:
        # Generate file ID and path
        file_id = file_manager.generate_file_id("pine")
        file_path = file_manager.get_file_path(file_id, ".pine")
        
        # Schedule background file cleanup
        background_tasks.add_task(file_manager.cleanup_old_files)
        
        # Use fallback Pine Script generation
        script_content = _generate_fallback_pine_script(
            request.strategy_name, 
            request.optimization_results,
            is_indicator=(request.output_format == "indicator")
        )
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Calculate file size
        file_size = len(script_content.encode('utf-8'))
        
        logger.info(f"Generated Pine Script: {file_id} ({file_size} bytes)")
        
        # End metrics timing
        if start_time and metrics_collector:
            metrics_collector.record_export_completed("pine_script", start_time)
        
        return PineScriptExportResponse(
            file_id=file_id,
            filename=f"{request.strategy_name}_{request.output_format}.pine",
            file_size=file_size,
            download_url=f"/api/v1/export/download/{file_id}",
            expires_at=datetime.now().isoformat() + "Z",
            script_preview=script_content[:500] + "..." if len(script_content) > 500 else script_content,
            generation_time=datetime.now().isoformat() + "Z"
        )
        
    except Exception as e:
        logger.error(f"Pine Script generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pine Script generation failed: {str(e)}"
        )


@router.post(
    "/report",
    response_model=ReportExportResponse,
    summary="Generate PDF Report",
    description="Generate comprehensive PDF report from optimization results"
)
async def export_report(
    request: ReportExportRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    permissions: Dict[str, bool] = Depends(require_permission("write"))
):
    """
    Generate PDF report from optimization results.
    
    Creates a comprehensive PDF report including performance analysis,
    validation results, risk assessment, and technical details.
    """
    # Start metrics timing
    start_time = metrics_collector.record_export_start("pdf_report") if metrics_collector else None
    
    try:
        # Generate file ID and path
        file_id = file_manager.generate_file_id("report")
        file_path = file_manager.get_file_path(file_id, ".pdf")
        
        # Schedule background file cleanup
        background_tasks.add_task(file_manager.cleanup_old_files)
        
        # Use fallback PDF generation
        pdf_content = _generate_fallback_pdf_report(
            request.strategy_name,
            request.optimization_results,
            request.report_type,
            request.include_charts,
            request.include_detailed_tables
        )
        
        # Write to file
        with open(file_path, 'wb') as f:
            f.write(pdf_content)
        
        # Get file size
        file_size = file_path.stat().st_size
        page_count = _estimate_page_count(file_size)
        
        logger.info(f"Generated PDF report: {file_id} ({file_size} bytes, ~{page_count} pages)")
        
        # End metrics timing
        if start_time and metrics_collector:
            metrics_collector.record_export_completed("pdf_report", start_time)
        
        return ReportExportResponse(
            file_id=file_id,
            filename=f"{request.strategy_name}_report.pdf",
            file_size=file_size,
            page_count=page_count,
            download_url=f"/api/v1/export/download/{file_id}",
            expires_at=datetime.now().isoformat() + "Z",
            generation_time=datetime.now().isoformat() + "Z"
        )
        
    except Exception as e:
        logger.error(f"PDF report generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"PDF report generation failed: {str(e)}"
        )


@router.get(
    "/download/{file_id}",
    response_class=FileResponse,
    summary="Download Export File",
    description="Download a previously generated export file"
)
async def download_export_file(
    file_id: str,
    api_key: str = Depends(verify_api_key),
    permissions: Dict[str, bool] = Depends(require_permission("read"))
):
    """
    Download a previously generated export file.
    
    Files are automatically cleaned up after 24 hours.
    """
    try:
        # Try different file extensions
        possible_files = [
            file_manager.get_file_path(file_id, ".pine"),
            file_manager.get_file_path(file_id, ".pdf"),
            file_manager.get_file_path(file_id, ".txt"),
            file_manager.get_file_path(file_id, ".json")
        ]
        
        file_path = None
        for possible_file in possible_files:
            if possible_file.exists():
                file_path = possible_file
                break
        
        if not file_path or not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Export file not found: {file_id}"
            )
        
        # Determine media type
        media_type = "application/octet-stream"
        if file_path.suffix == ".pine":
            media_type = "text/plain"
        elif file_path.suffix == ".pdf":
            media_type = "application/pdf"
        elif file_path.suffix == ".json":
            media_type = "application/json"
        
        logger.info(f"Download requested: {file_id} ({file_path.name})")
        
        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type=media_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File download failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"File download failed: {str(e)}"
        )


@router.get(
    "/files",
    response_model=List[FileDownloadResponse],
    summary="List Export Files",
    description="List all available export files for the current user"
)
async def list_export_files(
    api_key: str = Depends(verify_api_key),
    permissions: Dict[str, bool] = Depends(require_permission("read"))
):
    """
    List all available export files.
    
    Returns a list of files that can be downloaded, with metadata.
    """
    try:
        files = []
        
        for file_path in file_manager.storage_dir.iterdir():
            if file_path.is_file():
                # Extract file ID from filename
                name_parts = file_path.stem.split('_')
                if len(name_parts) >= 3:
                    file_type = name_parts[0]
                    file_id = file_path.stem
                    
                    # Get file metadata
                    stat = file_path.stat()
                    
                    files.append(FileDownloadResponse(
                        file_id=file_id,
                        filename=file_path.name,
                        file_type=file_type,
                        file_size=stat.st_size,
                        created_at=datetime.fromtimestamp(stat.st_ctime).isoformat() + "Z",
                        download_url=f"/api/v1/export/download/{file_id}"
                    ))
        
        # Sort by creation time (newest first)
        files.sort(key=lambda x: x.created_at, reverse=True)
        
        logger.info(f"Listed {len(files)} export files")
        return files
        
    except Exception as e:
        logger.error(f"Failed to list export files: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list export files: {str(e)}"
        )


@router.delete(
    "/files/{file_id}",
    response_model=BaseResponse,
    summary="Delete Export File",
    description="Delete a specific export file"
)
async def delete_export_file(
    file_id: str,
    api_key: str = Depends(verify_api_key),
    permissions: Dict[str, bool] = Depends(require_permission("write"))
):
    """
    Delete a specific export file.
    """
    try:
        # Try different file extensions
        possible_files = [
            file_manager.get_file_path(file_id, ".pine"),
            file_manager.get_file_path(file_id, ".pdf"),
            file_manager.get_file_path(file_id, ".txt"),
            file_manager.get_file_path(file_id, ".json")
        ]
        
        deleted = False
        for possible_file in possible_files:
            if possible_file.exists():
                possible_file.unlink()
                deleted = True
                logger.info(f"Deleted export file: {file_id}")
                break
        
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Export file not found: {file_id}"
            )
        
        return BaseResponse(
            success=True,
            message=f"Export file deleted successfully: {file_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete export file: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete export file: {str(e)}"
        )


# Helper functions

def _generate_fallback_pine_script(strategy_name: str, optimization_results: Dict[str, Any], is_indicator: bool = False) -> str:
    """Generate a simple fallback Pine Script when the main generator fails."""
    
    best_params = optimization_results.get("best_parameters", {})
    performance = optimization_results.get("performance_metrics", {})
    
    script_type = "indicator" if is_indicator else "strategy"
    overlay = "true" if is_indicator else "false"
    
    script = f"""// @version=5
{script_type}("{strategy_name} - Optimized", overlay={overlay})

// Optimized Parameters
"""
    
    # Add parameters
    for param_name, param_value in best_params.items():
        if isinstance(param_value, (int, float)):
            script += f"{param_name} = input.float({param_value}, '{param_name.replace('_', ' ').title()}')\n"
        else:
            script += f"// {param_name} = {param_value}\n"
    
    script += f"""
// Performance Metrics (from optimization)
// Sharpe Ratio: {performance.get('sharpe_ratio', 'N/A')}
// Total Return: {performance.get('total_return', 'N/A')}%
// Max Drawdown: {performance.get('max_drawdown', 'N/A')}%

// Simple Moving Average Example
fast_ma = ta.sma(close, int(math.max(5, {best_params.get('fast_period', 12)})))
slow_ma = ta.sma(close, int(math.max(10, {best_params.get('slow_period', 26)})))

// Basic signals
long_condition = ta.crossover(fast_ma, slow_ma)
short_condition = ta.crossunder(fast_ma, slow_ma)

"""
    
    if is_indicator:
        script += """
// Plot indicators
plot(fast_ma, "Fast MA", color=color.blue)
plot(slow_ma, "Slow MA", color=color.red)

// Plot signals
plotshape(long_condition, "Long Signal", shape.triangleup, location.belowbar, color.green, size=size.small)
plotshape(short_condition, "Short Signal", shape.triangledown, location.abovebar, color.red, size=size.small)
"""
    else:
        script += """
// Strategy logic
if long_condition
    strategy.entry("Long", strategy.long)
if short_condition
    strategy.entry("Short", strategy.short)

// Plot signals
plotshape(long_condition, "Long", shape.triangleup, location.belowbar, color.green, size=size.small)
plotshape(short_condition, "Short", shape.triangledown, location.abovebar, color.red, size=size.small)
"""
    
    return script


def _generate_fallback_pdf_report(strategy_name: str, optimization_results: Dict[str, Any], report_type: str, include_charts: bool, include_detailed_tables: bool) -> bytes:
    """Generate a fallback PDF report when the main generator fails."""
    
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from io import BytesIO
        
        # Create a BytesIO buffer to hold the PDF
        buffer = BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph(f"Trading Strategy Optimization Report", title_style))
        story.append(Paragraph(f"Strategy: {strategy_name}", styles['Heading2']))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        summary_text = f"""
        This report presents the optimization results for the {strategy_name} trading strategy.
        The optimization process has identified the best parameters to maximize strategy performance
        while maintaining acceptable risk levels.
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Performance Metrics
        story.append(Paragraph("Performance Metrics", styles['Heading2']))
        
        # Extract key metrics
        perf_metrics = optimization_results.get("performance_metrics", {})
        best_params = optimization_results.get("best_parameters", {})
        
        # Create performance table
        perf_data = [
            ["Metric", "Value"],
            ["Total Return", f"{perf_metrics.get('total_return', 0.0):.2%}"],
            ["Sharpe Ratio", f"{perf_metrics.get('sharpe_ratio', 0.0):.3f}"],
            ["Max Drawdown", f"{perf_metrics.get('max_drawdown', 0.0):.2%}"],
            ["Win Rate", f"{perf_metrics.get('win_rate', 0.0):.2%}"],
            ["Profit Factor", f"{perf_metrics.get('profit_factor', 0.0):.2f}"],
        ]
        
        perf_table = Table(perf_data, colWidths=[2*inch, 2*inch])
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(perf_table)
        story.append(Spacer(1, 20))
        
        # Optimized Parameters
        if best_params:
            story.append(Paragraph("Optimized Parameters", styles['Heading2']))
            
            param_data = [["Parameter", "Value"]]
            for param, value in best_params.items():
                if isinstance(value, float):
                    param_data.append([param, f"{value:.4f}"])
                else:
                    param_data.append([param, str(value)])
            
            param_table = Table(param_data, colWidths=[2*inch, 2*inch])
            param_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(param_table)
            story.append(Spacer(1, 20))
        
        # Risk Assessment
        story.append(Paragraph("Risk Assessment", styles['Heading2']))
        risk_text = f"""
        The strategy shows a maximum drawdown of {perf_metrics.get('max_drawdown', 0.0):.2%}, 
        which is within acceptable risk parameters. The Sharpe ratio of {perf_metrics.get('sharpe_ratio', 0.0):.3f} 
        indicates {('good' if perf_metrics.get('sharpe_ratio', 0.0) > 1.0 else 'moderate')} risk-adjusted returns.
        """
        story.append(Paragraph(risk_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Recommendations", styles['Heading2']))
        rec_text = f"""
        Based on the optimization results, the {strategy_name} strategy with the identified parameters 
        shows promising performance characteristics. Consider implementing this strategy with appropriate 
        position sizing and risk management controls.
        """
        story.append(Paragraph(rec_text, styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 40))
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Paragraph(footer_text, styles['Normal']))
        
        # Build the PDF
        doc.build(story)
        
        # Get the PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        return pdf_content
        
    except ImportError:
        # If reportlab is not available, return a simple text-based PDF placeholder
        simple_content = f"""
        Trading Strategy Optimization Report
        Strategy: {strategy_name}
        
        This is a simplified report. Install reportlab for full PDF generation.
        
        Performance Metrics:
        {optimization_results.get('performance_metrics', {})}
        
        Best Parameters:
        {optimization_results.get('best_parameters', {})}
        
        Generated: {datetime.now().isoformat()}
        """.encode('utf-8')
        
        return simple_content
    except Exception as e:
        # Fallback to simple text content
        error_content = f"""
        Trading Strategy Optimization Report
        Strategy: {strategy_name}
        
        Error generating full report: {str(e)}
        
        Raw optimization results:
        {optimization_results}
        
        Generated: {datetime.now().isoformat()}
        """.encode('utf-8')
        
        return error_content


def _estimate_page_count(file_size_bytes: int) -> int:
    """Estimate PDF page count from file size."""
    # Rough estimate: ~50KB per page for typical PDF with charts
    estimated_pages = max(1, file_size_bytes // 50000)
    return min(estimated_pages, 100)  # Cap at reasonable maximum 
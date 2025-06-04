"""
Report Templates

This module provides professional PDF templates for different sections of
trading strategy reports with consistent styling and layout.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image, KeepTogether, NextPageTemplate
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
import logging

try:
    from ..utils.logger import get_logger
except ImportError:
    from src.utils.logger import get_logger


@dataclass
class ReportStyling:
    """Professional styling configuration for reports."""
    
    # Colors (institutional blue theme)
    primary_color: colors.Color = colors.Color(0.1, 0.2, 0.4)  # Dark blue
    secondary_color: colors.Color = colors.Color(0.2, 0.4, 0.6)  # Medium blue
    accent_color: colors.Color = colors.Color(0.0, 0.5, 0.8)  # Light blue
    success_color: colors.Color = colors.Color(0.0, 0.6, 0.0)  # Green
    warning_color: colors.Color = colors.Color(0.8, 0.6, 0.0)  # Orange
    danger_color: colors.Color = colors.Color(0.8, 0.0, 0.0)  # Red
    neutral_color: colors.Color = colors.Color(0.5, 0.5, 0.5)  # Gray
    
    # Fonts
    title_font: str = 'Helvetica-Bold'
    heading_font: str = 'Helvetica-Bold'
    body_font: str = 'Helvetica'
    code_font: str = 'Courier'
    label_font: str = 'Helvetica'  # Added for chart compatibility
    
    # Font sizes
    title_size: int = 24
    heading1_size: int = 18
    heading2_size: int = 14
    heading3_size: int = 12
    body_size: int = 10
    caption_size: int = 8
    label_size: int = 10  # Added for chart compatibility
    
    # Spacing
    page_margin: float = 1.0 * inch
    section_spacing: float = 0.3 * inch
    paragraph_spacing: float = 0.15 * inch
    
    # Page settings
    page_size: Tuple[float, float] = A4
    
    # Chart settings
    chart_width: float = 6 * inch
    chart_height: float = 4 * inch
    small_chart_width: float = 4 * inch
    small_chart_height: float = 3 * inch
    
    def get_color_palette(self, n_colors: int = 8) -> List[colors.Color]:
        """Get a color palette for multiple series."""
        base_colors = [
            self.primary_color,
            self.secondary_color,
            self.accent_color,
            self.success_color,
            self.warning_color,
            self.danger_color,
            colors.Color(0.6, 0.2, 0.8),  # Purple
            colors.Color(0.8, 0.4, 0.2),  # Brown
        ]
        
        # Extend palette if needed
        while len(base_colors) < n_colors:
            base_colors.extend(base_colors)
        
        return base_colors[:n_colors]
    
    def get_paragraph_style(self, name: str) -> ParagraphStyle:
        """Get a paragraph style by name."""
        styles = {
            'Title': ParagraphStyle(
                'Title',
                fontName=self.title_font,
                fontSize=self.title_size,
                textColor=self.primary_color,
                alignment=TA_CENTER,
                spaceAfter=self.section_spacing
            ),
            'Heading1': ParagraphStyle(
                'Heading1',
                fontName=self.heading_font,
                fontSize=self.heading1_size,
                textColor=self.primary_color,
                spaceBefore=self.section_spacing,
                spaceAfter=self.paragraph_spacing
            ),
            'Heading2': ParagraphStyle(
                'Heading2',
                fontName=self.heading_font,
                fontSize=self.heading2_size,
                textColor=self.secondary_color,
                spaceBefore=self.paragraph_spacing,
                spaceAfter=self.paragraph_spacing
            ),
            'Heading3': ParagraphStyle(
                'Heading3',
                fontName=self.heading_font,
                fontSize=self.heading3_size,
                textColor=self.secondary_color,
                spaceBefore=self.paragraph_spacing,
                spaceAfter=self.paragraph_spacing * 0.5
            ),
            'Body': ParagraphStyle(
                'Body',
                fontName=self.body_font,
                fontSize=self.body_size,
                textColor=colors.black,
                alignment=TA_JUSTIFY,
                spaceAfter=self.paragraph_spacing
            ),
            'BodyCenter': ParagraphStyle(
                'BodyCenter',
                fontName=self.body_font,
                fontSize=self.body_size,
                textColor=colors.black,
                alignment=TA_CENTER,
                spaceAfter=self.paragraph_spacing
            ),
            'Caption': ParagraphStyle(
                'Caption',
                fontName=self.body_font,
                fontSize=self.caption_size,
                textColor=self.neutral_color,
                alignment=TA_CENTER,
                spaceAfter=self.paragraph_spacing * 0.5
            ),
            'Code': ParagraphStyle(
                'Code',
                fontName=self.code_font,
                fontSize=self.body_size - 1,
                textColor=colors.black,
                leftIndent=0.2 * inch,
                spaceAfter=self.paragraph_spacing
            )
        }
        return styles.get(name, styles['Body'])


class ReportPageTemplate:
    """Custom page template with headers and footers."""
    
    def __init__(self, styling: ReportStyling):
        self.styling = styling
        self.logger = get_logger("report_page_template")
    
    def draw_header(self, canvas: canvas.Canvas, doc, title: str = "Trading Strategy Report"):
        """Draw page header."""
        canvas.saveState()
        
        # Header line
        canvas.setStrokeColor(self.styling.primary_color)
        canvas.setLineWidth(2)
        canvas.line(
            self.styling.page_margin,
            doc.height + self.styling.page_margin - 0.5 * inch,
            doc.width + self.styling.page_margin,
            doc.height + self.styling.page_margin - 0.5 * inch
        )
        
        # Title
        canvas.setFont(self.styling.heading_font, 12)
        canvas.setFillColor(self.styling.primary_color)
        canvas.drawString(
            self.styling.page_margin,
            doc.height + self.styling.page_margin - 0.3 * inch,
            title
        )
        
        # Date
        canvas.setFont(self.styling.body_font, 10)
        canvas.setFillColor(self.styling.neutral_color)
        canvas.drawRightString(
            doc.width + self.styling.page_margin,
            doc.height + self.styling.page_margin - 0.3 * inch,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        
        canvas.restoreState()
    
    def draw_footer(self, canvas: canvas.Canvas, doc):
        """Draw page footer."""
        canvas.saveState()
        
        # Footer line
        canvas.setStrokeColor(self.styling.primary_color)
        canvas.setLineWidth(1)
        canvas.line(
            self.styling.page_margin,
            0.5 * inch,
            doc.width + self.styling.page_margin,
            0.5 * inch
        )
        
        # Page number
        canvas.setFont(self.styling.body_font, 10)
        canvas.setFillColor(self.styling.neutral_color)
        canvas.drawCentredString(
            doc.width / 2 + self.styling.page_margin,
            0.3 * inch,
            f"Page {doc.page}"
        )
        
        # Footer text
        canvas.drawString(
            self.styling.page_margin,
            0.3 * inch,
            "AI Trading Strategy Optimization System"
        )
        
        canvas.restoreState()


class ReportTemplate(ABC):
    """Base class for all report templates."""
    
    def __init__(self, styling: ReportStyling = None):
        """
        Initialize the report template.
        
        Args:
            styling: Report styling configuration
        """
        self.styling = styling or ReportStyling()
        self.page_template = ReportPageTemplate(self.styling)
        self.logger = get_logger(f"report_template_{self.__class__.__name__}")
        
        # Content storage
        self.content: List[Any] = []
        self.toc_entries: List[Tuple[str, int]] = []
    
    @abstractmethod
    def generate_content(self, data: Dict[str, Any]) -> List[Any]:
        """
        Generate template content from data.
        
        Args:
            data: Data dictionary containing all required information
            
        Returns:
            List of ReportLab flowables
        """
        pass
    
    def add_title(self, title: str) -> Paragraph:
        """Add a title to the content."""
        return Paragraph(title, self.styling.get_paragraph_style('Title'))
    
    def add_heading(self, text: str, level: int = 1) -> Paragraph:
        """Add a heading to the content."""
        style_name = f'Heading{level}'
        return Paragraph(text, self.styling.get_paragraph_style(style_name))
    
    def add_paragraph(self, text: str, style: str = 'Body') -> Paragraph:
        """Add a paragraph to the content."""
        return Paragraph(text, self.styling.get_paragraph_style(style))
    
    def add_spacer(self, height: float = None) -> Spacer:
        """Add vertical spacing."""
        height = height or self.styling.section_spacing
        return Spacer(1, height)
    
    def create_table(
        self,
        data: List[List[str]],
        headers: List[str] = None,
        col_widths: List[float] = None,
        style_name: str = 'default'
    ) -> Table:
        """Create a formatted table."""
        
        # Prepare data
        table_data = []
        if headers:
            table_data.append(headers)
        table_data.extend(data)
        
        # Create table
        table = Table(table_data, colWidths=col_widths)
        
        # Apply styling
        if style_name == 'default':
            table_style = TableStyle([
                # Header styling
                ('BACKGROUND', (0, 0), (-1, 0), self.styling.primary_color),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), self.styling.heading_font),
                ('FONTSIZE', (0, 0), (-1, 0), self.styling.body_size),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                
                # Body styling
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), self.styling.body_font),
                ('FONTSIZE', (0, 1), (-1, -1), self.styling.body_size),
                ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
                
                # Grid
                ('GRID', (0, 0), (-1, -1), 1, self.styling.neutral_color),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                
                # Alternating row colors
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)])
            ])
        elif style_name == 'metrics':
            table_style = TableStyle([
                # Metric name column
                ('BACKGROUND', (0, 0), (0, -1), self.styling.secondary_color),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
                ('FONTNAME', (0, 0), (0, -1), self.styling.heading_font),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                
                # Value column
                ('BACKGROUND', (1, 0), (-1, -1), colors.white),
                ('TEXTCOLOR', (1, 0), (-1, -1), colors.black),
                ('FONTNAME', (1, 0), (-1, -1), self.styling.body_font),
                ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                
                # Grid and spacing
                ('GRID', (0, 0), (-1, -1), 1, self.styling.neutral_color),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 0), (-1, -1), self.styling.body_size)
            ])
        
        table.setStyle(table_style)
        return table
    
    def format_percentage(self, value: float, decimals: int = 2) -> str:
        """Format a value as percentage."""
        if pd.isna(value) or value is None:
            return "N/A"
        return f"{value * 100:.{decimals}f}%"
    
    def format_currency(self, value: float, decimals: int = 2) -> str:
        """Format a value as currency."""
        if pd.isna(value) or value is None:
            return "N/A"
        return f"${value:,.{decimals}f}"
    
    def format_number(self, value: float, decimals: int = 2) -> str:
        """Format a number with specified decimals."""
        if pd.isna(value) or value is None:
            return "N/A"
        return f"{value:.{decimals}f}"


class ExecutiveSummaryTemplate(ReportTemplate):
    """Template for executive summary section."""
    
    def generate_content(self, data: Dict[str, Any]) -> List[Any]:
        """Generate executive summary content."""
        content = []
        
        # Title
        content.append(self.add_title("Executive Summary"))
        content.append(self.add_spacer())
        
        # Strategy overview
        strategy_name = data.get('strategy_name', 'Unknown Strategy')
        content.append(self.add_heading("Strategy Overview", 2))
        content.append(self.add_paragraph(
            f"This report presents a comprehensive analysis of the <b>{strategy_name}</b> "
            f"trading strategy, including performance metrics, validation results, and risk assessments."
        ))
        
        # Key metrics summary
        content.append(self.add_heading("Key Performance Metrics", 2))
        
        performance_data = data.get('performance_metrics', {})
        key_metrics = [
            ['Metric', 'Value'],
            ['Total Return', self.format_percentage(performance_data.get('total_return', 0))],
            ['Annual Return', self.format_percentage(performance_data.get('annual_return', 0))],
            ['Sharpe Ratio', self.format_number(performance_data.get('sharpe_ratio', 0))],
            ['Maximum Drawdown', self.format_percentage(performance_data.get('max_drawdown', 0))],
            ['Win Rate', self.format_percentage(performance_data.get('win_rate', 0))],
            ['Profit Factor', self.format_number(performance_data.get('profit_factor', 0))]
        ]
        
        content.append(self.create_table(
            key_metrics[1:], 
            headers=key_metrics[0], 
            col_widths=[3*inch, 2*inch],
            style_name='metrics'
        ))
        content.append(self.add_spacer())
        
        # Risk assessment summary
        content.append(self.add_heading("Risk Assessment", 2))
        
        risk_data = data.get('risk_assessment', {})
        overall_risk = risk_data.get('overall_risk_level', 'Medium')
        risk_score = risk_data.get('risk_score', 50)
        
        content.append(self.add_paragraph(
            f"The strategy has been assessed with an overall risk level of <b>{overall_risk}</b> "
            f"(Risk Score: {risk_score}/100). This assessment considers overfitting risk, "
            f"performance consistency, and validation results across multiple market conditions."
        ))
        
        # Validation summary
        content.append(self.add_heading("Validation Results", 2))
        
        validation_data = data.get('validation_results', {})
        validation_score = validation_data.get('overall_score', 0)
        deployment_status = validation_data.get('deployment_recommendation', 'Under Review')
        
        content.append(self.add_paragraph(
            f"The strategy achieved a validation score of <b>{validation_score:.1f}/100</b> "
            f"across multiple testing frameworks. The deployment recommendation is: <b>{deployment_status}</b>."
        ))
        
        # Recommendations
        content.append(self.add_heading("Key Recommendations", 2))
        
        recommendations = data.get('recommendations', [])
        if recommendations:
            rec_text = "<br/>".join([f"• {rec}" for rec in recommendations[:5]])
            content.append(self.add_paragraph(rec_text))
        else:
            content.append(self.add_paragraph("No specific recommendations at this time."))
        
        return content


class PerformanceAnalysisTemplate(ReportTemplate):
    """Template for detailed performance analysis section."""
    
    def generate_content(self, data: Dict[str, Any]) -> List[Any]:
        """Generate performance analysis content."""
        content = []
        
        # Title
        content.append(self.add_title("Performance Analysis"))
        content.append(self.add_spacer())
        
        # Core performance metrics
        content.append(self.add_heading("Core Performance Metrics", 2))
        
        performance_data = data.get('performance_metrics', {})
        
        # Primary metrics table
        primary_metrics = [
            ['Metric', 'Value', 'Description'],
            ['Total Return', 
             self.format_percentage(performance_data.get('total_return', 0)),
             'Cumulative return over the entire period'],
            ['Annual Return', 
             self.format_percentage(performance_data.get('annual_return', 0)),
             'Annualized return'],
            ['Volatility', 
             self.format_percentage(performance_data.get('volatility', 0)),
             'Annualized standard deviation of returns'],
            ['Sharpe Ratio', 
             self.format_number(performance_data.get('sharpe_ratio', 0)),
             'Risk-adjusted return measure'],
            ['Sortino Ratio', 
             self.format_number(performance_data.get('sortino_ratio', 0)),
             'Downside risk-adjusted return'],
            ['Calmar Ratio', 
             self.format_number(performance_data.get('calmar_ratio', 0)),
             'Return to maximum drawdown ratio']
        ]
        
        content.append(self.create_table(
            primary_metrics[1:], 
            headers=primary_metrics[0], 
            col_widths=[2*inch, 1.5*inch, 3*inch]
        ))
        content.append(self.add_spacer())
        
        # Risk metrics
        content.append(self.add_heading("Risk Metrics", 2))
        
        risk_metrics = [
            ['Metric', 'Value', 'Description'],
            ['Maximum Drawdown', 
             self.format_percentage(performance_data.get('max_drawdown', 0)),
             'Largest peak-to-trough decline'],
            ['Average Drawdown', 
             self.format_percentage(performance_data.get('avg_drawdown', 0)),
             'Average of all drawdown periods'],
            ['Recovery Factor', 
             self.format_number(performance_data.get('recovery_factor', 0)),
             'Net profit to maximum drawdown ratio'],
            ['VaR (95%)', 
             self.format_percentage(performance_data.get('var_95', 0)),
             'Value at Risk at 95% confidence'],
            ['CVaR (95%)', 
             self.format_percentage(performance_data.get('cvar_95', 0)),
             'Conditional Value at Risk']
        ]
        
        content.append(self.create_table(
            risk_metrics[1:], 
            headers=risk_metrics[0], 
            col_widths=[2*inch, 1.5*inch, 3*inch]
        ))
        content.append(self.add_spacer())
        
        # Trading metrics
        content.append(self.add_heading("Trading Activity Metrics", 2))
        
        trading_data = data.get('trading_metrics', {})
        trading_metrics = [
            ['Metric', 'Value', 'Description'],
            ['Total Trades', 
             str(trading_data.get('total_trades', 0)),
             'Total number of completed trades'],
            ['Win Rate', 
             self.format_percentage(trading_data.get('win_rate', 0)),
             'Percentage of profitable trades'],
            ['Profit Factor', 
             self.format_number(trading_data.get('profit_factor', 0)),
             'Gross profit to gross loss ratio'],
            ['Average Trade', 
             self.format_currency(trading_data.get('avg_trade', 0)),
             'Average profit/loss per trade'],
            ['Average Win', 
             self.format_currency(trading_data.get('avg_win', 0)),
             'Average profit of winning trades'],
            ['Average Loss', 
             self.format_currency(trading_data.get('avg_loss', 0)),
             'Average loss of losing trades'],
            ['Expectancy', 
             self.format_currency(trading_data.get('expectancy', 0)),
             'Expected value per trade']
        ]
        
        content.append(self.create_table(
            trading_metrics[1:], 
            headers=trading_metrics[0], 
            col_widths=[2*inch, 1.5*inch, 3*inch]
        ))
        content.append(self.add_spacer())
        
        # Performance attribution
        content.append(self.add_heading("Performance Attribution", 2))
        
        attribution_text = (
            "The strategy's performance can be attributed to several key factors:\n\n"
            "• <b>Market Timing:</b> The strategy's ability to enter and exit positions at optimal times\n"
            "• <b>Risk Management:</b> Effective position sizing and stop-loss implementation\n"
            "• <b>Market Selection:</b> Focus on liquid, high-volume trading instruments\n"
            "• <b>Parameter Optimization:</b> Well-tuned parameters that adapt to market conditions"
        )
        content.append(self.add_paragraph(attribution_text))
        
        return content


class ValidationResultsTemplate(ReportTemplate):
    """Template for validation results section."""
    
    def generate_content(self, data: Dict[str, Any]) -> List[Any]:
        """Generate validation results content."""
        content = []
        
        # Title
        content.append(self.add_title("Validation Results"))
        content.append(self.add_spacer())
        
        # Overview
        content.append(self.add_heading("Validation Framework Overview", 2))
        content.append(self.add_paragraph(
            "The strategy has been subjected to a comprehensive validation framework "
            "designed to assess robustness, consistency, and deployment readiness. "
            "This includes multi-period testing, cross-asset validation, and anti-overfitting measures."
        ))
        content.append(self.add_spacer())
        
        # Multi-period validation
        content.append(self.add_heading("Multi-Period Validation", 2))
        
        multi_period_data = data.get('multi_period_validation', {})
        validation_score = multi_period_data.get('validation_score', 0)
        consistency_score = multi_period_data.get('consistency_score', 0)
        
        content.append(self.add_paragraph(
            f"The strategy achieved a multi-period validation score of <b>{validation_score:.1f}/100</b> "
            f"with a consistency score of <b>{consistency_score:.1f}/100</b> across different market conditions."
        ))
        
        # Period performance table
        period_results = multi_period_data.get('period_performances', [])
        if period_results:
            period_table = [['Period Type', 'Return', 'Sharpe Ratio', 'Max Drawdown', 'Trades']]
            for period in period_results[:10]:  # Show top 10 periods
                period_table.append([
                    period.get('period_type', 'Unknown'),
                    self.format_percentage(period.get('total_return', 0)),
                    self.format_number(period.get('sharpe_ratio', 0)),
                    self.format_percentage(period.get('max_drawdown', 0)),
                    str(period.get('total_trades', 0))
                ])
            
            content.append(self.create_table(
                period_table[1:], 
                headers=period_table[0], 
                col_widths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch]
            ))
        
        content.append(self.add_spacer())
        
        # Cross-asset validation
        content.append(self.add_heading("Cross-Asset Validation", 2))
        
        cross_asset_data = data.get('cross_asset_validation', {})
        asset_consistency = cross_asset_data.get('consistency_score', 0)
        
        content.append(self.add_paragraph(
            f"Cross-asset testing achieved a consistency score of <b>{asset_consistency:.1f}/100</b>, "
            f"indicating the strategy's ability to generalize across different but correlated assets."
        ))
        
        # Asset performance table
        asset_results = cross_asset_data.get('asset_results', [])
        if asset_results:
            asset_table = [['Asset', 'Return', 'Sharpe Ratio', 'Correlation']]
            for asset in asset_results[:8]:  # Show top 8 assets
                asset_table.append([
                    asset.get('asset_symbol', 'Unknown'),
                    self.format_percentage(asset.get('total_return', 0)),
                    self.format_number(asset.get('sharpe_ratio', 0)),
                    self.format_number(asset.get('correlation', 0))
                ])
            
            content.append(self.create_table(
                asset_table[1:], 
                headers=asset_table[0], 
                col_widths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch]
            ))
        
        content.append(self.add_spacer())
        
        # Regime analysis
        content.append(self.add_heading("Market Regime Analysis", 2))
        
        regime_data = data.get('regime_analysis', {})
        regime_consistency = regime_data.get('regime_consistency', 0)
        
        content.append(self.add_paragraph(
            f"The strategy demonstrated <b>{regime_consistency:.1f}/100</b> consistency "
            f"across different market regimes (bull, bear, sideways markets)."
        ))
        
        # Regime performance table
        regime_performances = regime_data.get('regime_performances', [])
        if regime_performances:
            regime_table = [['Market Regime', 'Return', 'Sharpe Ratio', 'Win Rate']]
            for regime in regime_performances:
                regime_table.append([
                    regime.get('regime', 'Unknown'),
                    self.format_percentage(regime.get('total_return', 0)),
                    self.format_number(regime.get('sharpe_ratio', 0)),
                    self.format_percentage(regime.get('win_rate', 0))
                ])
            
            content.append(self.create_table(
                regime_table[1:], 
                headers=regime_table[0], 
                col_widths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch]
            ))
        
        return content


class RiskAssessmentTemplate(ReportTemplate):
    """Template for risk assessment section."""
    
    def generate_content(self, data: Dict[str, Any]) -> List[Any]:
        """Generate risk assessment content."""
        content = []
        
        # Title
        content.append(self.add_title("Risk Assessment"))
        content.append(self.add_spacer())
        
        # Overall risk summary
        content.append(self.add_heading("Overall Risk Assessment", 2))
        
        risk_data = data.get('risk_assessment', {})
        overall_risk = risk_data.get('overall_risk_level', 'Medium')
        risk_score = risk_data.get('risk_score', 50)
        
        content.append(self.add_paragraph(
            f"The strategy has been assigned an overall risk level of <b>{overall_risk}</b> "
            f"with a comprehensive risk score of <b>{risk_score}/100</b>. This assessment "
            f"considers multiple risk factors including overfitting, market dependency, "
            f"and performance consistency."
        ))
        content.append(self.add_spacer())
        
        # Anti-overfitting analysis
        content.append(self.add_heading("Anti-Overfitting Analysis", 2))
        
        overfitting_data = data.get('anti_overfitting', {})
        
        # Overfitting risk table
        overfitting_metrics = [
            ['Risk Factor', 'Score', 'Level', 'Description'],
            ['Data Mining Risk', 
             f"{overfitting_data.get('data_mining_risk', 0)}/100",
             self._get_risk_level(overfitting_data.get('data_mining_risk', 0)),
             'Risk of strategy being a statistical artifact'],
            ['Overfitting Risk', 
             f"{overfitting_data.get('overfitting_risk', 0)}/100",
             self._get_risk_level(overfitting_data.get('overfitting_risk', 0)),
             'Risk of curve fitting to historical data'],
            ['Parameter Complexity', 
             f"{overfitting_data.get('complexity_score', 0)}/100",
             self._get_risk_level(overfitting_data.get('complexity_score', 0)),
             'Strategy complexity and parameter count'],
            ['Persistence Score', 
             f"{overfitting_data.get('persistence_score', 0)}/100",
             self._get_risk_level(100 - overfitting_data.get('persistence_score', 0)),
             'Consistency across different conditions']
        ]
        
        content.append(self.create_table(
            overfitting_metrics[1:], 
            headers=overfitting_metrics[0], 
            col_widths=[1.8*inch, 0.8*inch, 0.8*inch, 2.6*inch]
        ))
        content.append(self.add_spacer())
        
        # Market dependency analysis
        content.append(self.add_heading("Market Dependency Analysis", 2))
        
        dependency_data = data.get('market_dependency', {})
        
        content.append(self.add_paragraph(
            "The strategy's dependency on specific market conditions has been analyzed "
            "to assess its robustness across different market environments."
        ))
        
        # Market correlation table
        correlations = dependency_data.get('market_correlations', {})
        correlation_table = [
            ['Market Factor', 'Correlation', 'Dependency Level'],
            ['Market Direction', 
             self.format_number(correlations.get('market_direction', 0)),
             self._get_dependency_level(abs(correlations.get('market_direction', 0)))],
            ['Volatility', 
             self.format_number(correlations.get('volatility', 0)),
             self._get_dependency_level(abs(correlations.get('volatility', 0)))],
            ['Volume', 
             self.format_number(correlations.get('volume', 0)),
             self._get_dependency_level(abs(correlations.get('volume', 0)))],
            ['Sector Performance', 
             self.format_number(correlations.get('sector', 0)),
             self._get_dependency_level(abs(correlations.get('sector', 0)))]
        ]
        
        content.append(self.create_table(
            correlation_table[1:], 
            headers=correlation_table[0], 
            col_widths=[2*inch, 1.5*inch, 2*inch]
        ))
        content.append(self.add_spacer())
        
        # Risk mitigation recommendations
        content.append(self.add_heading("Risk Mitigation Recommendations", 2))
        
        recommendations = risk_data.get('mitigation_recommendations', [])
        if recommendations:
            rec_text = "<br/>".join([f"• {rec}" for rec in recommendations])
            content.append(self.add_paragraph(rec_text))
        else:
            content.append(self.add_paragraph(
                "• Continue monitoring strategy performance across different market conditions<br/>"
                "• Implement position sizing based on volatility<br/>"
                "• Regular revalidation of strategy parameters<br/>"
                "• Diversification across multiple strategies"
            ))
        
        # Risk monitoring framework
        content.append(self.add_heading("Ongoing Risk Monitoring", 2))
        content.append(self.add_paragraph(
            "A continuous risk monitoring framework should be implemented to track "
            "the strategy's performance and risk characteristics over time. Key metrics "
            "to monitor include rolling Sharpe ratio, drawdown duration, and correlation "
            "with market factors."
        ))
        
        return content
    
    def _get_risk_level(self, score: float) -> str:
        """Convert risk score to risk level."""
        if score < 30:
            return "Low"
        elif score < 60:
            return "Medium"
        else:
            return "High"
    
    def _get_dependency_level(self, correlation: float) -> str:
        """Convert correlation to dependency level."""
        if correlation < 0.3:
            return "Low"
        elif correlation < 0.6:
            return "Medium"
        else:
            return "High"


class TechnicalAppendixTemplate(ReportTemplate):
    """Template for technical appendix section."""
    
    def generate_content(self, data: Dict[str, Any]) -> List[Any]:
        """Generate technical appendix content."""
        content = []
        
        # Title
        content.append(self.add_title("Technical Appendix"))
        content.append(self.add_spacer())
        
        # Strategy configuration
        content.append(self.add_heading("Strategy Configuration", 2))
        
        strategy_config = data.get('strategy_config', {})
        
        # Parameters table
        parameters = strategy_config.get('parameters', {})
        if parameters:
            param_table = [['Parameter', 'Value', 'Description']]
            for param_name, param_value in parameters.items():
                description = strategy_config.get('parameter_descriptions', {}).get(param_name, 'No description')
                param_table.append([
                    param_name,
                    str(param_value),
                    description
                ])
            
            content.append(self.create_table(
                param_table[1:], 
                headers=param_table[0], 
                col_widths=[2*inch, 1.5*inch, 3*inch]
            ))
        
        content.append(self.add_spacer())
        
        # Risk parameters
        content.append(self.add_heading("Risk Management Parameters", 2))
        
        risk_params = strategy_config.get('risk_params', {})
        if risk_params:
            risk_table = [['Parameter', 'Value', 'Description']]
            for param_name, param_value in risk_params.items():
                risk_table.append([
                    param_name,
                    str(param_value),
                    self._get_risk_param_description(param_name)
                ])
            
            content.append(self.create_table(
                risk_table[1:], 
                headers=risk_table[0], 
                col_widths=[2*inch, 1.5*inch, 3*inch]
            ))
        
        content.append(self.add_spacer())
        
        # Data specifications
        content.append(self.add_heading("Data Specifications", 2))
        
        data_specs = data.get('data_specifications', {})
        
        data_info = [
            ['Specification', 'Value'],
            ['Data Source', data_specs.get('source', 'Unknown')],
            ['Time Period', f"{data_specs.get('start_date', 'N/A')} to {data_specs.get('end_date', 'N/A')}"],
            ['Frequency', data_specs.get('frequency', 'Daily')],
            ['Total Records', str(data_specs.get('total_records', 0))],
            ['Assets Covered', str(data_specs.get('asset_count', 0))],
            ['Data Quality Score', f"{data_specs.get('quality_score', 0)}/100"]
        ]
        
        content.append(self.create_table(
            data_info[1:], 
            headers=data_info[0], 
            col_widths=[3*inch, 2*inch],
            style_name='metrics'
        ))
        content.append(self.add_spacer())
        
        # Optimization details
        content.append(self.add_heading("Optimization Details", 2))
        
        optimization_data = data.get('optimization_details', {})
        
        content.append(self.add_paragraph(
            f"The strategy was optimized using {optimization_data.get('method', 'Hyperopt')} "
            f"with {optimization_data.get('iterations', 0)} iterations. "
            f"The optimization process took {optimization_data.get('duration_hours', 0):.1f} hours "
            f"and evaluated {optimization_data.get('parameter_combinations', 0)} parameter combinations."
        ))
        
        # Validation methodology
        content.append(self.add_heading("Validation Methodology", 2))
        
        validation_method = data.get('validation_methodology', {})
        
        content.append(self.add_paragraph(
            "The strategy validation employed a multi-layered approach including:"
        ))
        
        methodology_text = (
            "• <b>Data Splitting:</b> 80/20 train/test split with walk-forward analysis<br/>"
            "• <b>Cross-Asset Testing:</b> Validation across correlated instruments<br/>"
            "• <b>Multi-Period Analysis:</b> Testing across different market regimes<br/>"
            "• <b>Anti-Overfitting Measures:</b> Parameter constraints and complexity penalties<br/>"
            "• <b>Statistical Testing:</b> Significance tests for performance differences"
        )
        content.append(self.add_paragraph(methodology_text))
        
        # Implementation notes
        content.append(self.add_heading("Implementation Notes", 2))
        
        impl_notes = data.get('implementation_notes', [])
        if impl_notes:
            notes_text = "<br/>".join([f"• {note}" for note in impl_notes])
            content.append(self.add_paragraph(notes_text))
        else:
            content.append(self.add_paragraph(
                "• Strategy implemented using Python with pandas and numpy<br/>"
                "• Backtesting engine handles realistic transaction costs and slippage<br/>"
                "• Risk management integrated at the position level<br/>"
                "• Performance metrics calculated using industry-standard methods"
            ))
        
        return content
    
    def _get_risk_param_description(self, param_name: str) -> str:
        """Get description for risk parameter."""
        descriptions = {
            'max_position_size': 'Maximum position size as percentage of capital',
            'stop_loss': 'Stop loss percentage',
            'take_profit': 'Take profit percentage',
            'max_drawdown_limit': 'Maximum allowed drawdown before stopping',
            'position_sizing_method': 'Method used for position sizing',
            'risk_per_trade': 'Risk percentage per individual trade'
        }
        return descriptions.get(param_name, 'Risk management parameter') 
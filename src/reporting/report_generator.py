"""
PDF Report Generator

This module provides the main ReportGenerator class that compiles all report
components (templates, data, visualizations) into professional PDF documents.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import io
import logging

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image, KeepTogether, NextPageTemplate, PageTemplate,
    BaseDocTemplate, Frame
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing

try:
    from .templates import (
        ReportTemplate, ReportStyling, ReportPageTemplate,
        ExecutiveSummaryTemplate, PerformanceAnalysisTemplate,
        ValidationResultsTemplate, RiskAssessmentTemplate,
        TechnicalAppendixTemplate
    )
    from .data_integration import ReportDataPackage, ReportDataCollector
    from .visualization import ReportVisualizationEngine
    from ..utils.logger import get_logger
except ImportError:
    from src.reporting.templates import (
        ReportTemplate, ReportStyling, ReportPageTemplate,
        ExecutiveSummaryTemplate, PerformanceAnalysisTemplate,
        ValidationResultsTemplate, RiskAssessmentTemplate,
        TechnicalAppendixTemplate
    )
    from src.reporting.data_integration import ReportDataPackage, ReportDataCollector
    from src.reporting.visualization import ReportVisualizationEngine
    from src.utils.logger import get_logger


@dataclass
class ReportConfiguration:
    """Configuration for report generation."""
    
    # Report content sections
    include_executive_summary: bool = True
    include_performance_analysis: bool = True
    include_validation_results: bool = True
    include_risk_assessment: bool = True
    include_technical_appendix: bool = True
    
    # Report format settings
    page_size: Tuple[float, float] = A4
    margins: float = 1.0 * inch
    generate_toc: bool = True
    
    # Content options
    include_charts: bool = True
    include_detailed_tables: bool = True
    include_code_snippets: bool = False
    
    # Output settings
    output_directory: str = "reports/output"
    filename_template: str = "{strategy_name}_report_{timestamp}.pdf"
    
    # Advanced options
    max_chart_size: Tuple[float, float] = (6*inch, 4*inch)
    chart_dpi: int = 300
    compress_images: bool = True


class CustomTableOfContents(TableOfContents):
    """Custom Table of Contents with professional styling."""
    
    def __init__(self, styling: ReportStyling):
        super().__init__()
        self.styling = styling
        
        # TOC styling
        self.levelStyles = [
            ParagraphStyle(
                'TOC1',
                fontName=styling.heading_font,
                fontSize=styling.heading2_size,
                textColor=styling.primary_color,
                leftIndent=0,
                spaceAfter=styling.paragraph_spacing
            ),
            ParagraphStyle(
                'TOC2',
                fontName=styling.body_font,
                fontSize=styling.body_size,
                textColor=styling.secondary_color,
                leftIndent=20,
                spaceAfter=styling.paragraph_spacing * 0.5
            ),
            ParagraphStyle(
                'TOC3',
                fontName=styling.body_font,
                fontSize=styling.body_size - 1,
                textColor=colors.black,
                leftIndent=40,
                spaceAfter=styling.paragraph_spacing * 0.3
            )
        ]


class ReportCanvas(canvas.Canvas):
    """Custom canvas for page templates and advanced features."""
    
    def __init__(self, *args, **kwargs):
        self.styling = kwargs.pop('styling', ReportStyling())
        self.report_title = kwargs.pop('report_title', 'Trading Strategy Report')
        super().__init__(*args, **kwargs)
        
        self._page_count = 0
        self.page_template = ReportPageTemplate(self.styling)
    
    def showPage(self):
        """Override showPage to add headers and footers."""
        self._page_count += 1
        
        # Skip header/footer on first page (title page)
        if self._page_count > 1:
            # Create a mock doc object for template compatibility
            class MockDoc:
                def __init__(self, width, height, page_num, styling):
                    self.width = width - 2 * styling.page_margin
                    self.height = height - 2 * styling.page_margin
                    self.page = page_num
            
            mock_doc = MockDoc(self._pagesize[0], self._pagesize[1], self._page_count, self.styling)
            
            # Draw header and footer
            self.page_template.draw_header(self, mock_doc, self.report_title)
            self.page_template.draw_footer(self, mock_doc)
        
        super().showPage()


class ReportGenerator:
    """Main PDF report generator that compiles all components."""
    
    def __init__(
        self,
        styling: ReportStyling = None,
        visualization_engine: ReportVisualizationEngine = None,
        config: ReportConfiguration = None
    ):
        """
        Initialize the report generator.
        
        Args:
            styling: Report styling configuration
            visualization_engine: Visualization engine for charts
            config: Report generation configuration
        """
        self.styling = styling or ReportStyling()
        self.visualization_engine = visualization_engine or ReportVisualizationEngine(self.styling)
        self.config = config or ReportConfiguration()
        self.logger = get_logger("report_generator")
        
        # Create output directory
        Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize templates
        self.templates = {
            'executive_summary': ExecutiveSummaryTemplate(self.styling),
            'performance_analysis': PerformanceAnalysisTemplate(self.styling),
            'validation_results': ValidationResultsTemplate(self.styling),
            'risk_assessment': RiskAssessmentTemplate(self.styling),
            'technical_appendix': TechnicalAppendixTemplate(self.styling)
        }
        
        # Document components
        self.content_elements = []
        self.toc_entries = []
    
    def generate_report(
        self,
        data_package: ReportDataPackage,
        equity_data: pd.Series = None,
        returns_data: pd.Series = None,
        output_filename: str = None
    ) -> str:
        """
        Generate a complete PDF report.
        
        Args:
            data_package: Complete data package for the report
            equity_data: Optional equity curve data for charts
            returns_data: Optional returns data for analysis
            output_filename: Optional custom output filename
            
        Returns:
            Path to the generated PDF file
        """
        self.logger.info(f"Generating PDF report for strategy: {data_package.strategy_name}")
        
        # Generate filename if not provided
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = self.config.filename_template.format(
                strategy_name=data_package.strategy_name.replace(' ', '_'),
                timestamp=timestamp
            )
        
        output_path = Path(self.config.output_directory) / output_filename
        
        try:
            # Generate all visualizations
            all_charts = self.visualization_engine.generate_all_charts(
                data_package, equity_data, returns_data
            )
            
            # Create PDF document
            doc = self._create_document(str(output_path), data_package.strategy_name)
            
            # Build content
            story = self._build_document_content(data_package, all_charts)
            
            # Generate PDF
            doc.build(story, canvasmaker=lambda *args, **kwargs: ReportCanvas(
                *args, 
                **kwargs, 
                styling=self.styling,
                report_title=f"{data_package.strategy_name} - Trading Strategy Report"
            ))
            
            self.logger.info(f"Report generated successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            raise
    
    def generate_executive_summary_only(
        self,
        data_package: ReportDataPackage,
        output_filename: str = None
    ) -> str:
        """Generate an executive summary report only."""
        
        # Temporarily modify config
        original_config = self.config
        self.config = ReportConfiguration(
            include_executive_summary=True,
            include_performance_analysis=False,
            include_validation_results=False,
            include_risk_assessment=False,
            include_technical_appendix=False,
            output_directory=original_config.output_directory
        )
        
        try:
            if not output_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{data_package.strategy_name.replace(' ', '_')}_executive_summary_{timestamp}.pdf"
            
            return self.generate_report(data_package, output_filename=output_filename)
        finally:
            self.config = original_config
    
    def _create_document(self, filename: str, title: str) -> BaseDocTemplate:
        """Create the base document with page templates."""
        
        doc = BaseDocTemplate(
            filename,
            pagesize=self.config.page_size,
            leftMargin=self.config.margins,
            rightMargin=self.config.margins,
            topMargin=self.config.margins + 0.5*inch,  # Extra space for header
            bottomMargin=self.config.margins + 0.5*inch,  # Extra space for footer
            title=f"{title} - Trading Strategy Report",
            author="AI Trading Strategy Optimization System"
        )
        
        # Create frames for content
        frame = Frame(
            self.config.margins,
            self.config.margins,
            doc.width,
            doc.height,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0
        )
        
        # Create page templates
        normal_page_template = PageTemplate(
            id='normal',
            frames=[frame],
            pagesize=self.config.page_size
        )
        
        doc.addPageTemplates([normal_page_template])
        
        return doc
    
    def _build_document_content(
        self,
        data_package: ReportDataPackage,
        charts: Dict[str, Dict[str, Any]]
    ) -> List[Any]:
        """Build the complete document content."""
        
        story = []
        
        # Title page
        story.extend(self._create_title_page(data_package))
        story.append(PageBreak())
        
        # Table of contents (if enabled)
        if self.config.generate_toc:
            story.extend(self._create_table_of_contents())
            story.append(PageBreak())
        
        # Executive Summary
        if self.config.include_executive_summary:
            story.extend(self._add_section(
                'executive_summary',
                data_package,
                charts.get('executive_summary', {})
            ))
            story.append(PageBreak())
        
        # Performance Analysis
        if self.config.include_performance_analysis:
            story.extend(self._add_section(
                'performance_analysis',
                data_package,
                charts.get('performance_analysis', {})
            ))
            story.append(PageBreak())
        
        # Validation Results
        if self.config.include_validation_results:
            story.extend(self._add_section(
                'validation_results',
                data_package,
                charts.get('validation_results', {})
            ))
            story.append(PageBreak())
        
        # Risk Assessment
        if self.config.include_risk_assessment:
            story.extend(self._add_section(
                'risk_assessment',
                data_package,
                charts.get('risk_assessment', {})
            ))
            story.append(PageBreak())
        
        # Technical Appendix
        if self.config.include_technical_appendix:
            story.extend(self._add_section(
                'technical_appendix',
                data_package,
                {}
            ))
        
        return story
    
    def _create_title_page(self, data_package: ReportDataPackage) -> List[Any]:
        """Create the title page."""
        
        title_content = []
        
        # Main title
        title_content.append(Spacer(1, 2*inch))
        title_content.append(Paragraph(
            "Trading Strategy Analysis Report",
            self.styling.get_paragraph_style('Title')
        ))
        
        title_content.append(Spacer(1, 0.5*inch))
        
        # Strategy name
        title_content.append(Paragraph(
            f"<b>{data_package.strategy_name}</b>",
            ParagraphStyle(
                'StrategyTitle',
                fontName=self.styling.heading_font,
                fontSize=self.styling.heading1_size,
                textColor=self.styling.secondary_color,
                alignment=TA_CENTER,
                spaceAfter=self.styling.section_spacing
            )
        ))
        
        title_content.append(Spacer(1, 1*inch))
        
        # Summary information
        summary_info = [
            ['Report Generated:', data_package.generation_timestamp.strftime('%Y-%m-%d %H:%M:%S')],
            ['Overall Risk Level:', data_package.risk_assessment.get('overall_risk_level', 'N/A')],
            ['Validation Score:', f"{data_package.validation_results.get('overall_score', 0):.1f}/100"],
            ['Deployment Status:', data_package.validation_results.get('deployment_recommendation', 'Under Review')]
        ]
        
        summary_table = Table(summary_info, colWidths=[2.5*inch, 2.5*inch])
        summary_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), self.styling.body_font),
            ('FONTSIZE', (0, 0), (-1, -1), self.styling.body_size),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, self.styling.neutral_color),
            ('BACKGROUND', (0, 0), (0, -1), self.styling.secondary_color),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
            ('FONTNAME', (0, 0), (0, -1), self.styling.heading_font)
        ]))
        
        title_content.append(KeepTogether([
            Paragraph("Report Summary", self.styling.get_paragraph_style('Heading2')),
            Spacer(1, 0.2*inch),
            summary_table
        ]))
        
        title_content.append(Spacer(1, 1*inch))
        
        # Disclaimer
        disclaimer_text = (
            "<b>Disclaimer:</b> This report is generated by an automated trading strategy "
            "optimization system. Past performance does not guarantee future results. "
            "Trading involves substantial risk of loss and is not suitable for all investors. "
            "Please consult with a qualified financial advisor before making investment decisions."
        )
        
        title_content.append(Paragraph(
            disclaimer_text,
            ParagraphStyle(
                'Disclaimer',
                fontName=self.styling.body_font,
                fontSize=self.styling.caption_size,
                textColor=self.styling.neutral_color,
                alignment=TA_JUSTIFY,
                leftIndent=0.5*inch,
                rightIndent=0.5*inch
            )
        ))
        
        return title_content
    
    def _create_table_of_contents(self) -> List[Any]:
        """Create table of contents."""
        
        toc_content = []
        
        # TOC title
        toc_content.append(Paragraph(
            "Table of Contents",
            self.styling.get_paragraph_style('Title')
        ))
        toc_content.append(Spacer(1, 0.5*inch))
        
        # TOC entries
        toc_entries = []
        
        if self.config.include_executive_summary:
            toc_entries.append(('Executive Summary', 1))
        
        if self.config.include_performance_analysis:
            toc_entries.append(('Performance Analysis', 1))
        
        if self.config.include_validation_results:
            toc_entries.append(('Validation Results', 1))
        
        if self.config.include_risk_assessment:
            toc_entries.append(('Risk Assessment', 1))
        
        if self.config.include_technical_appendix:
            toc_entries.append(('Technical Appendix', 1))
        
        # Create TOC manually (simplified version)
        for entry, level in toc_entries:
            style = ParagraphStyle(
                'TOCEntry',
                fontName=self.styling.heading_font if level == 1 else self.styling.body_font,
                fontSize=self.styling.body_size + (2 if level == 1 else 0),
                textColor=self.styling.primary_color if level == 1 else colors.black,
                leftIndent=(level - 1) * 20,
                spaceAfter=self.styling.paragraph_spacing
            )
            toc_content.append(Paragraph(f"• {entry}", style))
        
        return toc_content
    
    def _add_section(
        self,
        section_name: str,
        data_package: ReportDataPackage,
        charts: Dict[str, Any]
    ) -> List[Any]:
        """Add a report section with content and charts."""
        
        content = []
        
        # Generate template content
        template = self.templates[section_name]
        template_content = template.generate_content(data_package.__dict__)
        content.extend(template_content)
        
        # Add charts if available and enabled
        if self.config.include_charts and charts:
            content.append(Spacer(1, self.styling.section_spacing))
            content.extend(self._embed_charts(charts))
        
        return content
    
    def _embed_charts(self, charts: Dict[str, Any]) -> List[Any]:
        """Embed charts into the document."""
        
        chart_elements = []
        
        for chart_name, chart in charts.items():
            try:
                # Convert chart to image
                chart_image = self.visualization_engine.convert_chart_to_image(chart)
                
                if chart_image:
                    # Add chart with caption
                    chart_elements.append(KeepTogether([
                        chart_image,
                        Spacer(1, 0.1*inch),
                        Paragraph(
                            f"Figure: {chart_name.replace('_', ' ').title()}",
                            self.styling.get_paragraph_style('Caption')
                        ),
                        Spacer(1, self.styling.paragraph_spacing)
                    ]))
                
            except Exception as e:
                self.logger.warning(f"Failed to embed chart {chart_name}: {e}")
        
        return chart_elements
    
    def generate_batch_reports(
        self,
        data_packages: List[ReportDataPackage],
        output_directory: str = None
    ) -> List[str]:
        """Generate reports for multiple strategies."""
        
        if output_directory:
            original_dir = self.config.output_directory
            self.config.output_directory = output_directory
        
        generated_files = []
        
        try:
            for data_package in data_packages:
                try:
                    output_file = self.generate_report(data_package)
                    generated_files.append(output_file)
                    self.logger.info(f"Generated report for {data_package.strategy_name}")
                except Exception as e:
                    self.logger.error(f"Failed to generate report for {data_package.strategy_name}: {e}")
            
            self.logger.info(f"Batch generation completed. Generated {len(generated_files)} reports.")
            return generated_files
            
        finally:
            if output_directory:
                self.config.output_directory = original_dir
    
    def generate_comparison_report(
        self,
        data_packages: List[ReportDataPackage],
        output_filename: str = None
    ) -> str:
        """Generate a comparison report for multiple strategies."""
        
        # This would be implemented to compare multiple strategies
        # For now, we'll create a placeholder that generates individual reports
        self.logger.info("Comparison report functionality not yet implemented")
        
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"strategy_comparison_{timestamp}.pdf"
        
        # For now, generate individual reports
        return self.generate_batch_reports(data_packages) 
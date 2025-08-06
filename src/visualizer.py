"""Visualization utilities for traffic analysis and predictions."""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from prophet import Prophet
from .config import Config


class TrafficVisualizer:
    """Creates visualizations for traffic analysis and predictions."""
    
    def __init__(self, output_dir: Path, config: Optional[Config] = None):
        """
        Initialize visualizer with output directory and configuration.
        
        Args:
            output_dir: Directory to save visualization files
            config: Configuration object with export and styling options
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.config = config or Config()
        
        # Set plotly theme from config
        pio.templates.default = self.config.plotly_theme
        self.colors = px.colors.qualitative.Set2
    
    def _export_figure(self, fig: go.Figure, filename: str, **kwargs) -> None:
        """
        Export figure in multiple formats based on configuration.
        
        Args:
            fig: Plotly figure to export
            filename: Base filename (without extension)
            **kwargs: Additional arguments for specific export formats
        """
        base_path = self.output_dir / filename
        exported_formats = []
        
        # Export HTML (interactive)
        if self.config.export_html:
            html_path = base_path.with_suffix('.html')
            fig.write_html(str(html_path))
            exported_formats.append('HTML')
        
        # Export PNG (static, high quality)
        if self.config.export_png:
            png_path = base_path.with_suffix('.png')
            width = kwargs.get('width', self.config.chart_width)
            height = kwargs.get('height', self.config.chart_height)
            fig.write_image(
                str(png_path), 
                width=width, 
                height=height,
                scale=self.config.image_scale
            )
            exported_formats.append('PNG')
        
        # Export SVG (vector graphics)
        if self.config.export_svg:
            svg_path = base_path.with_suffix('.svg')
            width = kwargs.get('width', self.config.chart_width)
            height = kwargs.get('height', self.config.chart_height)
            fig.write_image(str(svg_path), width=width, height=height)
            exported_formats.append('SVG')
        
        # Export PDF (print-ready)
        if self.config.export_pdf:
            pdf_path = base_path.with_suffix('.pdf')
            width = kwargs.get('width', self.config.chart_width)
            height = kwargs.get('height', self.config.chart_height)
            fig.write_image(str(pdf_path), width=width, height=height)
            exported_formats.append('PDF')
        
        if exported_formats:
            print(f"Exported {filename} as: {', '.join(exported_formats)}")
    
    def plot_forecast(self, df: pd.DataFrame, forecast: pd.DataFrame, 
                     model: Prophet) -> None:
        """
        Create forecast visualization.
        
        Args:
            df: Historical data
            forecast: Prophet forecast DataFrame
            model: Fitted Prophet model
        """
        fig = go.Figure()
        
        # Add historical data with enhanced hover
        fig.add_trace(go.Scatter(
            x=df['ds'], y=df['y'],
            mode='markers',
            name='Historical Data',
            marker=dict(color=self.colors[0], size=4, opacity=0.6),
            hovertemplate='<b>Historical Data</b><br>' +
                         'Date: %{x}<br>' +
                         'Clicks: %{y:,.0f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add forecast line with enhanced hover
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color=self.colors[1], width=2),
            hovertemplate='<b>Forecast</b><br>' +
                         'Date: %{x}<br>' +
                         'Predicted: %{y:,.0f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(68, 68, 68, 0.2)',
            name='Confidence Interval',
            hovertemplate='<b>Confidence Interval</b><br>' +
                         'Date: %{x}<br>' +
                         'Lower: %{y:,.0f}<br>' +
                         'Upper: %{customdata:,.0f}<br>' +
                         '<extra></extra>',
            customdata=forecast['yhat_upper']
        ))
        
        # Add forecast start line
        max_historical = df['ds'].max()
        fig.add_shape(
            type="line",
            x0=max_historical, x1=max_historical,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="red", dash="dash", width=2),
        )
        
        # Add annotation for forecast start
        fig.add_annotation(
            x=max_historical,
            y=0.95,
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1
        )
        
        fig.update_layout(
            title='Organic Traffic Forecast',
            xaxis_title='Date',
            yaxis_title='Daily Clicks',
            width=self.config.chart_width,
            height=self.config.chart_height,
            hovermode='x unified',
            # Enhanced interactivity
            dragmode='zoom',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add range selector buttons
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=30, label="30d", step="day", stepmode="backward"),
                        dict(count=90, label="90d", step="day", stepmode="backward"),
                        dict(count=180, label="6m", step="day", stepmode="backward"),
                        dict(count=365, label="1y", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        self._export_figure(fig, 'forecast')
    
    def create_interactive_dashboard(self, df: pd.DataFrame, forecast: pd.DataFrame, 
                                   analysis: Dict[str, Any]) -> None:
        """
        Create an interactive dashboard with dropdown filters and toggles.
        
        Args:
            df: Historical data
            forecast: Prophet forecast DataFrame
            analysis: Analysis results dictionary
        """
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=df['ds'], y=df['y'],
            mode='markers',
            name='Historical Data',
            visible=True,
            marker=dict(color=self.colors[0], size=4, opacity=0.6),
            hovertemplate='<b>Historical</b><br>Date: %{x}<br>Clicks: %{y:,.0f}<extra></extra>'
        ))
        
        # Add forecast data
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            visible=True,
            line=dict(color=self.colors[1], width=2),
            hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Predicted: %{y:,.0f}<extra></extra>'
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            visible=True,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(68, 68, 68, 0.2)',
            name='Confidence Interval',
            visible=True,
            hovertemplate='<b>Confidence</b><br>Date: %{x}<br>Range: %{y:,.0f} - %{customdata:,.0f}<extra></extra>',
            customdata=forecast['yhat_upper']
        ))
        
        # Add trend line
        if 'trend' in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['trend'],
                mode='lines',
                name='Trend Only',
                visible=False,
                line=dict(color=self.colors[3], width=2, dash='dash'),
                hovertemplate='<b>Trend</b><br>Date: %{x}<br>Value: %{y:,.0f}<extra></extra>'
            ))
        
        # Create interactive controls
        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    x=0.1, y=1.15,
                    buttons=list([
                        dict(label="All Data", method="update", 
                             args=[{"visible": [True, True, True, True, False]}]),
                        dict(label="Historical Only", method="update",
                             args=[{"visible": [True, False, False, False, False]}]),
                        dict(label="Forecast Only", method="update",
                             args=[{"visible": [False, True, True, True, False]}]),
                        dict(label="Trend Analysis", method="update",
                             args=[{"visible": [True, False, False, False, True]}])
                    ])
                )
            ]
        )
        
        fig.update_layout(
            title=dict(
                text='Interactive Traffic Dashboard<br><sub>Use dropdown to customize view</sub>',
                x=0.5, font_size=16
            ),
            xaxis_title='Date',
            yaxis_title='Daily Clicks',
            width=self.config.dashboard_width,
            height=self.config.dashboard_height,
            hovermode='x unified',
            dragmode='zoom',
            margin=dict(t=120),
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="7d", step="day", stepmode="backward"),
                        dict(count=30, label="30d", step="day", stepmode="backward"),
                        dict(count=90, label="90d", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True, thickness=0.05),
                type="date"
            )
        )
        
        if self.config.create_dashboard:
            # Only export HTML for dashboard (interactive features)
            html_path = self.output_dir / 'dashboard.html'
            fig.write_html(str(html_path))
            print(f"Interactive dashboard created: {html_path}")
        else:
            print("Dashboard creation disabled in configuration")
    
    def plot_components(self, model: Prophet, forecast: pd.DataFrame) -> None:
        """
        Plot model components.
        
        Args:
            model: Fitted Prophet model
            forecast: Prophet forecast DataFrame
        """
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Trend', 'Weekly Seasonality', 'Yearly Seasonality'],
            vertical_spacing=0.08
        )
        
        # Trend component with enhanced hover
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['trend'],
            mode='lines',
            name='Trend',
            line=dict(color=self.colors[0], width=2),
            hovertemplate='<b>Trend</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:,.1f}<br>' +
                         '<extra></extra>'
        ), row=1, col=1)
        
        # Weekly component with enhanced hover
        if 'weekly' in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['weekly'],
                mode='lines',
                name='Weekly',
                line=dict(color=self.colors[1], width=2),
                hovertemplate='<b>Weekly Seasonality</b><br>' +
                             'Date: %{x}<br>' +
                             'Effect: %{y:,.1f}<br>' +
                             '<extra></extra>'
            ), row=2, col=1)
        
        # Yearly component with enhanced hover
        if 'yearly' in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yearly'],
                mode='lines',
                name='Yearly',
                line=dict(color=self.colors[2], width=2),
                hovertemplate='<b>Yearly Seasonality</b><br>' +
                             'Date: %{x}<br>' +
                             'Effect: %{y:,.1f}<br>' +
                             '<extra></extra>'
            ), row=3, col=1)
        
        fig.update_layout(
            title='Traffic Patterns Decomposition',
            width=self.config.chart_width,
            height=int(self.config.chart_height * 1.3),  # Taller for 3 subplots
            showlegend=True,
            hovermode='x unified',
            dragmode='zoom',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add synchronized zooming across subplots
        fig.update_xaxes(matches='x')
        
        # Add range slider to bottom subplot
        fig.update_layout(
            xaxis3=dict(
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        self._export_figure(fig, 'components', height=int(self.config.chart_height * 1.3))
    
    def plot_analysis(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> None:
        """
        Create comprehensive analysis visualizations.
        
        Args:
            df: Traffic data
            analysis: Analysis results dictionary
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Weekly Pattern', 'Monthly Pattern', 'Growth Trend', 'Traffic Distribution'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        self._plot_weekly_pattern_plotly(fig, analysis['weekly_pattern'], row=1, col=1)
        self._plot_monthly_pattern_plotly(fig, analysis['monthly_pattern'], row=1, col=2)
        self._plot_growth_trend_plotly(fig, df, row=2, col=1)
        self._plot_distribution_plotly(fig, df, row=2, col=2)
        
        fig.update_layout(
            title='Traffic Analysis Overview',
            width=int(self.config.chart_width * 1.2),  # Wider for 2x2 layout
            height=int(self.config.chart_height * 1.3),
            showlegend=True,
            hovermode='closest',
            dragmode='zoom'
        )
        
        # Add dropdown for metric selection (will be enhanced in future)
        fig.update_layout(
            annotations=[
                dict(
                    text="💡 Tip: Use toolbar to zoom, pan, and reset views. Hover for detailed info.",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.05,
                    xanchor='center', yanchor='top',
                    font=dict(size=12, color="#666")
                )
            ]
        )
        
        self._export_figure(fig, 'analysis', 
                          width=int(self.config.chart_width * 1.2),
                          height=int(self.config.chart_height * 1.3))
    
    def _plot_weekly_pattern_plotly(self, fig, weekly_data: pd.DataFrame, row: int, col: int) -> None:
        """Plot weekly traffic pattern."""
        colors = ['green' if strength > 0 else 'red' if abs(strength) > 5 else 'blue' 
                 for strength in weekly_data['relative_strength']]
        
        fig.add_trace(go.Bar(
            x=weekly_data['day'], 
            y=weekly_data['average_clicks'],
            marker_color=colors,
            name='Weekly Pattern',
            text=[f'{strength:+.1f}%' for strength in weekly_data['relative_strength']],
            textposition='outside',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>' +
                         'Avg Clicks: %{y:,.0f}<br>' +
                         'vs Average: %{text}<br>' +
                         '<extra></extra>'
        ), row=row, col=col)
        
        fig.update_xaxes(title_text='Day of Week', row=row, col=col)
        fig.update_yaxes(title_text='Average Clicks', row=row, col=col)
    
    def _plot_monthly_pattern_plotly(self, fig, monthly_data: pd.DataFrame, row: int, col: int) -> None:
        """Plot monthly traffic pattern by year-month."""
        peak_month = monthly_data.loc[monthly_data['average_clicks'].idxmax()]
        
        fig.add_trace(go.Scatter(
            x=monthly_data['year_month'], 
            y=monthly_data['average_clicks'],
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=8),
            name='Monthly Pattern',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>' +
                         'Avg Clicks: %{y:,.0f}<br>' +
                         'vs Average: %{customdata:+.1f}%<br>' +
                         '<extra></extra>',
            customdata=monthly_data['relative_strength']
        ), row=row, col=col)
        
        # Add peak indicator in annotation
        fig.add_annotation(
            x=peak_month['year_month'],
            y=peak_month['average_clicks'],
            text=f"Peak: {peak_month['year_month']}",
            showarrow=True,
            arrowhead=2,
            row=row, col=col
        )
        
        fig.update_xaxes(title_text='Year-Month', row=row, col=col, tickangle=45)
        fig.update_yaxes(title_text='Average Clicks', row=row, col=col)
    
    def _plot_growth_trend_plotly(self, fig, df: pd.DataFrame, row: int, col: int) -> None:
        """Plot growth trend over time."""
        # Daily data points with enhanced hover
        fig.add_trace(go.Scatter(
            x=df['ds'], y=df['y'],
            mode='markers',
            marker=dict(size=4, opacity=0.3),
            name='Daily',
            showlegend=False,
            hovertemplate='<b>Daily Traffic</b><br>' +
                         'Date: %{x}<br>' +
                         'Clicks: %{y:,.0f}<br>' +
                         '<extra></extra>'
        ), row=row, col=col)
        
        # Monthly average
        df_monthly = df.groupby(df['ds'].dt.to_period('M')).agg({
            'ds': 'first',
            'y': 'mean'
        })
        fig.add_trace(go.Scatter(
            x=df_monthly['ds'], y=df_monthly['y'],
            mode='lines',
            line=dict(color='red', width=2),
            name='Monthly Average',
            showlegend=False
        ), row=row, col=col)
        
        # Trend line
        z = np.polyfit(df.index, df['y'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=df['ds'], y=p(df.index),
            mode='lines',
            line=dict(color='green', dash='dash', width=2),
            name='Trend',
            showlegend=False
        ), row=row, col=col)
        
        fig.update_xaxes(title_text='Date', row=row, col=col)
        fig.update_yaxes(title_text='Clicks', row=row, col=col)
    
    def _plot_distribution_plotly(self, fig, df: pd.DataFrame, row: int, col: int) -> None:
        """Plot traffic distribution."""
        mean_val = df['y'].mean()
        median_val = df['y'].median()
        
        fig.add_trace(go.Histogram(
            x=df['y'],
            nbinsx=50,
            opacity=0.7,
            name='Distribution',
            showlegend=False,
            hovertemplate='<b>Traffic Distribution</b><br>' +
                         'Range: %{x}<br>' +
                         'Count: %{y}<br>' +
                         '<extra></extra>'
        ), row=row, col=col)
        
        # Mean and median info will be available via hover tooltips in Phase 2
        
        fig.update_xaxes(title_text='Daily Clicks', row=row, col=col)
        fig.update_yaxes(title_text='Frequency', row=row, col=col)
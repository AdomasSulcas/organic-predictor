"""Configuration settings for the Traffic Prophet model."""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Config:
    """Configuration for Prophet model and analysis."""
    
    # Prophet model parameters
    growth: str = 'linear'
    changepoint_prior_scale: float = 0.05
    changepoint_range: float = 0.9
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = False
    seasonality_mode: str = 'multiplicative'
    interval_width: float = 0.95
    uncertainty_samples: int = 1000
    
    monthly_seasonality_period: float = 30.5
    monthly_seasonality_fourier: int = 5
    
    validation_days: int = 60
    
    # Data column mappings
    date_column: str = 'Date'
    clicks_column: str = 'Clicks'
    impressions_column: str = 'Impressions'
    ctr_column: str = 'CTR'
    position_column: str = 'Position'
    
    # Visualization output options
    export_html: bool = True
    export_png: bool = True
    export_svg: bool = False
    export_pdf: bool = False
    create_dashboard: bool = True
    
    # Chart dimensions
    chart_width: int = 1000
    chart_height: int = 600
    dashboard_width: int = 1200
    dashboard_height: int = 700
    
    # Image export quality
    image_scale: float = 2.0  # For high DPI displays
    
    # Theme and styling
    plotly_theme: str = 'plotly_white'  # plotly_white, plotly_dark, ggplot2, seaborn, etc.
    
    @property
    def prophet_params(self) -> dict:
        """Get parameters for Prophet initialization."""
        return {
            'growth': self.growth,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'changepoint_range': self.changepoint_range,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'seasonality_mode': self.seasonality_mode,
            'interval_width': self.interval_width,
            'uncertainty_samples': self.uncertainty_samples,
        }
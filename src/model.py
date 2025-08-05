"""Prophet model implementation for traffic forecasting."""

from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from .config import Config


class TrafficProphetModel:
    """Prophet-based model for organic traffic prediction."""
    
    def __init__(self, config: Config):
        """
        Initialize the Prophet model with configuration.
        
        Args:
            config: Model configuration settings
        """
        self.config = config
        self.model: Optional[Prophet] = None
        self.validation_metrics: Dict[str, float] = {}
        
    def fit(self, df: pd.DataFrame) -> 'TrafficProphetModel':
        """
        Fit the Prophet model on training data.
        
        Args:
            df: Preprocessed DataFrame with 'ds' and 'y' columns
            
        Returns:
            Self for method chaining
        """
        train_df, test_df = self._split_data(df)
        
        self.model = self._create_model()
        self.model.fit(train_df[['ds', 'y']])
        
        if len(test_df) > 0:
            self._validate(train_df, test_df)
        
        self.model = self._create_model()
        self.model.fit(df[['ds', 'y']])
        
        return self
    
    def predict(self, periods: int) -> pd.DataFrame:
        """
        Generate future predictions.
        
        Args:
            periods: Number of days to forecast
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        return forecast
    
    def get_future_predictions(self, forecast: pd.DataFrame, 
                             historical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract future predictions from forecast.
        
        Args:
            forecast: Full forecast DataFrame from Prophet
            historical_df: Historical data DataFrame
            
        Returns:
            DataFrame with future predictions only
        """
        max_historical_date = historical_df['ds'].max()
        future_forecast = forecast[forecast['ds'] > max_historical_date].copy()
        
        future_forecast['predicted'] = future_forecast['yhat'].round().astype(int)
        future_forecast['lower_bound'] = future_forecast['yhat_lower'].round().astype(int)
        future_forecast['upper_bound'] = future_forecast['yhat_upper'].round().astype(int)
        
        return future_forecast[['ds', 'predicted', 'lower_bound', 'upper_bound']]
    
    def cross_validate(self, initial_days: int = 365, 
                      period_days: int = 30, 
                      horizon_days: int = 30) -> pd.DataFrame:
        """
        Perform time series cross-validation.
        
        Args:
            initial_days: Initial training period
            period_days: Period between cutoff dates
            horizon_days: Forecast horizon
            
        Returns:
            DataFrame with cross-validation results
        """
        if self.model is None:
            raise ValueError("Model must be fitted before cross-validation")
        
        cv_results = cross_validation(
            self.model,
            initial=f'{initial_days} days',
            period=f'{period_days} days',
            horizon=f'{horizon_days} days'
        )
        
        metrics_df = performance_metrics(cv_results)
        
        print(f"Cross-validation MAPE: {metrics_df['mape'].mean()*100:.1f}%")
        print(f"Cross-validation Coverage: {metrics_df['coverage'].mean()*100:.1f}%")
        
        return cv_results
    
    def _create_model(self) -> Prophet:
        """Create a Prophet model with configuration settings."""
        model = Prophet(**self.config.prophet_params)
        
        model.add_seasonality(
            name='monthly',
            period=self.config.monthly_seasonality_period,
            fourier_order=self.config.monthly_seasonality_fourier
        )
        
        return model
    
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and validation sets."""
        train_size = len(df) - self.config.validation_days
        train_size = max(train_size, int(len(df) * 0.8))
        
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        return train_df, test_df
    
    def _validate(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Validate model on test set."""
        test_forecast = self.model.predict(test_df[['ds']])
        
        mae = np.mean(np.abs(test_forecast['yhat'] - test_df['y']))
        mape = mean_absolute_percentage_error(test_df['y'], test_forecast['yhat']) * 100
        rmse = np.sqrt(mean_squared_error(test_df['y'], test_forecast['yhat']))
        
        self.validation_metrics = {
            'mae': mae,
            'mape': mape,
            'rmse': rmse
        }
        
        print(f"Validation Metrics - MAE: {mae:.0f}, MAPE: {mape:.1f}%, RMSE: {rmse:.0f}")
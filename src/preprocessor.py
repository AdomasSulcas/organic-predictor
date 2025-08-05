"""Data preprocessing for Prophet model."""

from typing import Optional
import pandas as pd
import numpy as np


class TrafficPreprocessor:
    """Preprocesses traffic data for Prophet modeling."""
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw traffic data into Prophet-ready format.
        
        Args:
            df: Raw traffic DataFrame
            
        Returns:
            Processed DataFrame with required columns and features
        """
        df_processed = df.copy()
        
        df_processed['ds'] = pd.to_datetime(df_processed['Date'])
        df_processed['y'] = df_processed['Clicks']
        
        df_processed['ctr'] = pd.to_numeric(
            df_processed['CTR'].str.rstrip('%'), 
            errors='coerce'
        ) / 100
        df_processed['position'] = df_processed['Position']
        
        df_processed = df_processed.sort_values('ds').reset_index(drop=True)
        
        df_processed = self._add_features(df_processed)
        
        return df_processed
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based and derived features.
        
        Args:
            df: DataFrame with basic columns
            
        Returns:
            DataFrame with additional features
        """
        df['day_of_week'] = df['ds'].dt.dayofweek
        df['day_of_month'] = df['ds'].dt.day
        df['week_of_year'] = df['ds'].dt.isocalendar().week
        df['month'] = df['ds'].dt.month
        df['quarter'] = df['ds'].dt.quarter
        df['year'] = df['ds'].dt.year
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_end'] = df['ds'].dt.is_month_end.astype(int)
        df['is_month_start'] = df['ds'].dt.is_month_start.astype(int)
        
        df['clicks_per_impression'] = df['y'] / df['Impressions']
        df['position_impact'] = 1 / df['position']
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
        """
        Identify and optionally handle outliers.
        
        Args:
            df: DataFrame with traffic data
            z_threshold: Z-score threshold for outlier detection
            
        Returns:
            DataFrame with outliers marked
        """
        df = df.copy()
        z_scores = np.abs((df['y'] - df['y'].mean()) / df['y'].std())
        df['is_outlier'] = z_scores > z_threshold
        
        print(f"Identified {df['is_outlier'].sum()} outliers")
        return df
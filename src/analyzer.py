"""Traffic pattern analysis utilities."""

from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats


class TrafficAnalyzer:
    """Analyzes traffic patterns and provides insights."""
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive traffic analysis.
        
        Args:
            df: Preprocessed traffic DataFrame
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'basic_stats': self._calculate_basic_stats(df),
            'weekly_pattern': self._analyze_weekly_pattern(df),
            'monthly_pattern': self._analyze_monthly_pattern(df),
            'growth_metrics': self._calculate_growth_metrics(df),
            'anomalies': self._detect_anomalies(df),
            'seasonality_strength': self._calculate_seasonality_strength(df)
        }
        
        return results
    
    def _calculate_basic_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic statistical metrics."""
        return {
            'mean': df['y'].mean(),
            'median': df['y'].median(),
            'std': df['y'].std(),
            'min': df['y'].min(),
            'max': df['y'].max(),
            'cv': df['y'].std() / df['y'].mean()
        }
    
    def _analyze_weekly_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze weekly traffic patterns."""
        df['day_name'] = df['ds'].dt.day_name()
        weekly_avg = df.groupby('day_name')['y'].mean()
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                     'Friday', 'Saturday', 'Sunday']
        weekly_avg = weekly_avg.reindex(day_order)
        
        weekly_pattern = pd.DataFrame({
            'day': day_order,
            'average_clicks': weekly_avg.values,
            'relative_strength': (weekly_avg.values / weekly_avg.mean() - 1) * 100
        })
        
        return weekly_pattern
    
    def _analyze_monthly_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze monthly traffic patterns."""
        monthly_avg = df.groupby('month')['y'].mean()
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        monthly_pattern = pd.DataFrame({
            'month': month_names,
            'average_clicks': [monthly_avg.get(i+1, 0) for i in range(12)],
            'relative_strength': [(monthly_avg.get(i+1, 0) / monthly_avg.mean() - 1) * 100 
                                for i in range(12)]
        })
        
        return monthly_pattern
    
    def _calculate_growth_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate growth-related metrics."""
        first_30_days = df.head(30)['y'].mean()
        last_30_days = df.tail(30)['y'].mean()
        
        monthly_growth = df.groupby(df['ds'].dt.to_period('M'))['y'].mean()
        avg_monthly_growth = monthly_growth.pct_change().mean()
        
        return {
            'total_growth_pct': (last_30_days / first_30_days - 1) * 100,
            'avg_monthly_growth_pct': avg_monthly_growth * 100,
            'first_30_days_avg': first_30_days,
            'last_30_days_avg': last_30_days
        }
    
    def _detect_anomalies(self, df: pd.DataFrame, z_threshold: float = 3.0) -> List[Dict]:
        """Detect anomalous traffic days."""
        z_scores = np.abs(stats.zscore(df['y']))
        anomaly_mask = z_scores > z_threshold
        
        anomalies = []
        for idx in df[anomaly_mask].index:
            anomalies.append({
                'date': df.loc[idx, 'ds'],
                'clicks': df.loc[idx, 'y'],
                'z_score': z_scores[idx]
            })
        
        return anomalies
    
    def _calculate_seasonality_strength(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate the strength of different seasonality patterns."""
        weekly_var = df.groupby(df['ds'].dt.dayofweek)['y'].mean().var()
        monthly_var = df.groupby(df['ds'].dt.month)['y'].mean().var()
        total_var = df['y'].var()
        
        return {
            'weekly_strength': np.sqrt(weekly_var / total_var),
            'monthly_strength': np.sqrt(monthly_var / total_var)
        }
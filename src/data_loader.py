"""Data loading utilities for traffic data."""

from pathlib import Path
from typing import Union
import pandas as pd


class TrafficDataLoader:
    """Handles loading and initial validation of traffic data."""
    
    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize the data loader.
        
        Args:
            file_path: Path to the CSV file containing traffic data
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
    
    def load(self) -> pd.DataFrame:
        """
        Load traffic data from CSV file.
        
        Returns:
            DataFrame with traffic data
            
        Raises:
            ValueError: If required columns are missing
        """
        df = pd.read_csv(self.file_path)
        
        required_columns = ['Date', 'Clicks', 'Impressions', 'CTR', 'Position']
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"Loaded {len(df)} rows of traffic data")
        return df
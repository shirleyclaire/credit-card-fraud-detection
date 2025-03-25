import pandas as pd
import numpy as np
from datetime import datetime

class TimeFeatureGenerator:
    def __init__(self, base_datetime=None):
        # Set reference datetime (e.g., first transaction time)
        self.base_datetime = base_datetime or datetime(2023, 1, 1)
        self.unusual_hours = set(range(0, 6))  # 12AM to 6AM
        
    def _get_datetime(self, seconds):
        """Convert seconds elapsed to datetime"""
        return self.base_datetime + pd.Timedelta(seconds=seconds)
    
    def generate_features(self, df):
        """Generate all time-based features"""
        X = df.copy()
        
        # Convert seconds to datetime
        timestamps = X['Time'].apply(self._get_datetime)
        
        # Basic time unit features
        X['hour'] = timestamps.dt.hour
        X['day_of_week'] = timestamps.dt.dayofweek
        X['is_weekend'] = X['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical time features
        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
        X['day_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 7)
        X['day_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 7)
        
        # Unusual hours flag
        X['unusual_hour'] = X['hour'].isin(self.unusual_hours).astype(int)
        
        # Transaction velocity features (30-minute and 1-hour windows)
        X = X.sort_values('Time')
        windows = [1800, 3600]  # 30 minutes and 1 hour in seconds
        
        for window in windows:
            # Count transactions in rolling window
            tx_count = X.groupby(pd.Grouper(key='Time', freq=f'{window}S')).size()
            X[f'tx_count_{window}s'] = X['Time'].map(tx_count)
            
            # Calculate time since last transaction
            X[f'time_since_last_{window}s'] = X['Time'].diff()
        
        # Transaction pattern deviation features
        hour_avg = X.groupby('hour')['Amount'].transform('mean')
        X['amount_hour_deviation'] = (X['Amount'] - hour_avg) / hour_avg
        
        # Drop temporary columns
        X.drop(['hour', 'day_of_week'], axis=1, inplace=True)
        
        return X

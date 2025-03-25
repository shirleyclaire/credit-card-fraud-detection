import pandas as pd
import numpy as np
from scipy import stats

class TransactionPatternGenerator:
    def __init__(self, windows=[1800, 3600, 86400]):  # 30min, 1h, 24h in seconds
        self.windows = windows
        self.percentile_thresholds = [75, 90, 95, 99]
        
    def generate_features(self, df):
        """Generate transaction pattern features"""
        X = df.copy().sort_values('Time')
        
        # Initialize storage for rolling statistics
        for window in self.windows:
            X[f'amount_mean_{window}s'] = np.nan
            X[f'amount_std_{window}s'] = np.nan
            X[f'amount_min_{window}s'] = np.nan
            X[f'amount_max_{window}s'] = np.nan
            X[f'tx_count_{window}s'] = np.nan
            
        # Calculate rolling statistics for each window
        for window in self.windows:
            rolling_stats = X.rolling(
                window=pd.Timedelta(seconds=window), 
                on='Time'
            )['Amount'].agg(['mean', 'std', 'min', 'max', 'count'])
            
            X[f'amount_mean_{window}s'] = rolling_stats['mean'].shift(1)
            X[f'amount_std_{window}s'] = rolling_stats['std'].shift(1)
            X[f'amount_min_{window}s'] = rolling_stats['min'].shift(1)
            X[f'amount_max_{window}s'] = rolling_stats['max'].shift(1)
            X[f'tx_count_{window}s'] = rolling_stats['count'].shift(1)
            
            # Calculate deviation from moving average
            X[f'amount_deviation_{window}s'] = (
                X['Amount'] - X[f'amount_mean_{window}s']
            ) / (X[f'amount_std_{window}s'] + 1e-8)
            
            # Ratio to historical average
            X[f'amount_ratio_{window}s'] = (
                X['Amount'] / (X[f'amount_mean_{window}s'] + 1e-8)
            )
        
        # Global percentile features
        for pct in self.percentile_thresholds:
            threshold = np.percentile(X['Amount'], pct)
            X[f'above_{pct}th_percentile'] = (X['Amount'] > threshold).astype(int)
        
        # Sequence patterns
        X['amount_diff'] = X['Amount'].diff()
        X['amount_diff_pct'] = X['amount_diff'] / (X['Amount'].shift(1) + 1e-8)
        X['amount_acceleration'] = X['amount_diff'].diff()
        
        # Binary flags for unusual patterns
        X['is_amount_spike'] = (
            X['Amount'] > X['amount_mean_3600s'] + 2 * X['amount_std_3600s']
        ).astype(int)
        
        X['is_rapid_sequence'] = (
            X['tx_count_1800s'] > X['tx_count_1800s'].mean() + 2 * X['tx_count_1800s'].std()
        ).astype(int)
        
        # Handle missing values for first transactions
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(0)
        
        return X

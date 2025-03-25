import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def load_data(self):
        """Load and perform initial data cleaning"""
        df = pd.read_csv(self.file_path)
        
        # Handle missing values
        missing_stats = df.isnull().sum()
        if missing_stats.any():
            print("Missing value statistics:")
            print(missing_stats[missing_stats > 0])
            
            # Fill missing values appropriately
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        return df
    
    def split_data(self, df, test_size=0.2, random_state=42, n_splits=5):
        """Split data with option for stratified k-fold"""
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Create both train/test and stratified k-fold splits
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        # Create stratified k-fold indices for cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_indices = list(skf.split(X_train, y_train))
        
        return {
            'train_test': (X_train, X_test, y_train, y_test),
            'fold_indices': fold_indices,
            'full_data': (X, y)
        }

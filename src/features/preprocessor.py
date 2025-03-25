import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from .time_features import TimeFeatureGenerator
from .transaction_patterns import TransactionPatternGenerator
from .pca_features import PCAFeatureGenerator

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.amount_scaler = RobustScaler()
        self.time_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.time_generator = TimeFeatureGenerator()
        self.pattern_generator = TransactionPatternGenerator()
        self.pca_generator = PCAFeatureGenerator()
    
    def fit(self, X, y=None):
        """Fit all scalers"""
        if 'Amount' in X.columns:
            self.amount_scaler.fit(X[['Amount']])
        if 'Time' in X.columns:
            self.time_scaler.fit(X[['Time']])
            
        # Fit scaler for V1-V28 features
        v_features = [col for col in X.columns if col.startswith('V')]
        if v_features:
            self.feature_scaler.fit(X[v_features])
        return self
        
    def transform(self, X):
        """Apply transformations"""
        X_transformed = X.copy()
        
        # Generate transaction pattern features before scaling
        X_transformed = self.pattern_generator.generate_features(X_transformed)
        
        # Generate time features before scaling
        if 'Time' in X_transformed.columns:
            X_transformed = self.time_generator.generate_features(X_transformed)
            
        # Generate PCA-based features
        pca_features = self.pca_generator.generate_features(X_transformed)
        X_transformed = pd.concat([X_transformed, pca_features], axis=1)
        
        # Transform Amount using RobustScaler (handles outliers better)
        if 'Amount' in X_transformed.columns:
            X_transformed['Amount_Scaled'] = self.amount_scaler.transform(X_transformed[['Amount']])
            X_transformed.drop('Amount', axis=1, inplace=True)
            
        # Scale V1-V28 features if needed
        v_features = [col for col in X_transformed.columns if col.startswith('V')]
        if v_features:
            X_transformed[v_features] = self.feature_scaler.transform(X_transformed[v_features])
            
        return X_transformed

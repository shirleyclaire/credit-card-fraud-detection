import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, f1_score
import joblib
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

class FraudDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, contamination=0.002, n_estimators=100, max_samples='auto',
                 max_features=1.0, bootstrap=False, n_jobs=-1, threshold=None,
                 random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.threshold = threshold
        self.random_state = random_state
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
    
    def fit(self, X, y=None):
        """Train the Isolation Forest model"""
        # Use parallel processing for large datasets
        if X.shape[0] > 10000:
            with ThreadPoolExecutor() as executor:
                self.model.fit(X)
        else:
            self.model.fit(X)
        
        # Store decision function scores for threshold optimization
        self.decision_scores_ = self.model.decision_function(X)
        return self
    
    def predict(self, X):
        """Predict using optimal threshold if set, otherwise use default"""
        scores = self.decision_function(X)
        if self.threshold is not None:
            return (scores < self.threshold).astype(int)
        return (self.model.predict(X) == -1).astype(int)
    
    def decision_function(self, X):
        """Get anomaly scores"""
        return -self.model.decision_function(X)  # Negative to make higher score = more anomalous
    
    def optimize_threshold(self, X_val, y_val):
        """Find optimal threshold using precision-recall curve"""
        scores = self.decision_function(X_val)
        precisions, recalls, thresholds = precision_recall_curve(y_val, scores)
        
        # Calculate F1 scores for different thresholds
        f1_scores = np.zeros_like(thresholds)
        for i, threshold in enumerate(thresholds):
            predictions = (scores >= threshold).astype(int)
            f1_scores[i] = f1_score(y_val, predictions)
        
        # Find optimal threshold
        optimal_idx = np.argmax(f1_scores)
        self.threshold = thresholds[optimal_idx]
        
        # Plot precision-recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, label='PR curve')
        plt.scatter(recalls[optimal_idx], precisions[optimal_idx], 
                   color='red', label='Optimal threshold')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve with Optimal Threshold')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return {
            'optimal_threshold': self.threshold,
            'optimal_f1': f1_scores[optimal_idx],
            'precision': precisions[optimal_idx],
            'recall': recalls[optimal_idx]
        }
    
    def get_anomaly_factors(self, X):
        """Calculate normalized anomaly factors"""
        scores = self.decision_function(X)
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        return normalized_scores
    
    def save_model(self, path):
        """Save the model and its parameters"""
        model_data = {
            'model': self.model,
            'threshold': self.threshold,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load_model(cls, path):
        """Load a saved model"""
        model_data = joblib.load(path)
        detector = cls(
            contamination=model_data['contamination'],
            n_estimators=model_data['n_estimators']
        )
        detector.model = model_data['model']
        detector.threshold = model_data['threshold']
        return detector

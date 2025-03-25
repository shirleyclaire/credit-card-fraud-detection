import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from concurrent.futures import ThreadPoolExecutor
import joblib
from collections import defaultdict

class EnsembleFraudDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, contamination=0.002, n_jobs=-1, random_state=42):
        self.contamination = contamination
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Base models for heterogeneous ensemble
        self.base_models = {
            'iforest': IsolationForest(
                contamination=self.contamination,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            ),
            'ocsvm': OneClassSVM(
                kernel='rbf',
                nu=self.contamination
            ),
            'lof': LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True,
                n_jobs=self.n_jobs
            )
        }
        
        # Specialized models for different transaction segments
        self.segment_models = {}
        
        # Meta-learner for stacking
        self.meta_learner = LogisticRegression(
            class_weight='balanced',
            random_state=self.random_state
        )
        
        # Feature importance tracking
        self.feature_weights = None
        
        # Time window performance tracking
        self.time_weights = defaultdict(lambda: 1.0)
        self.performance_history = []
        
    def fit(self, X, y=None):
        """Fit ensemble components with parallel processing"""
        # Train base models
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                name: executor.submit(self._fit_model, model, X)
                for name, model in self.base_models.items()
            }
            
            for name, future in futures.items():
                self.base_models[name] = future.result()
        
        # Train segment-specific models
        self._train_segment_models(X, y)
        
        # Generate meta-features for stacking
        meta_features = self._generate_meta_features(X)
        
        # Train meta-learner if labels are provided
        if y is not None:
            self.meta_learner.fit(meta_features, y)
            
            # Calculate feature importance weights
            self.feature_weights = self._calculate_feature_weights(X, y)
        
        return self
    
    def predict(self, X):
        """Generate ensemble predictions"""
        # Get base model predictions
        base_predictions = self._get_base_predictions(X)
        
        # Get segment model predictions
        segment_predictions = self._get_segment_predictions(X)
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X)
        
        # Combine predictions using learned weights
        if hasattr(self.meta_learner, 'predict_proba'):
            ensemble_scores = self.meta_learner.predict_proba(meta_features)[:, 1]
        else:
            # Fallback to weighted voting
            predictions = np.column_stack([base_predictions, segment_predictions])
            ensemble_scores = np.average(predictions, weights=self._get_current_weights(), axis=1)
        
        return (ensemble_scores >= self.contamination).astype(int)
    
    def decision_function(self, X):
        """Generate anomaly scores"""
        meta_features = self._generate_meta_features(X)
        if hasattr(self.meta_learner, 'predict_proba'):
            return self.meta_learner.predict_proba(meta_features)[:, 1]
        return self._get_weighted_scores(X)
    
    def _fit_model(self, model, X):
        """Helper function for parallel model training"""
        return model.fit(X)
    
    def _train_segment_models(self, X, y=None):
        """Train specialized models for different transaction segments"""
        # Amount-based segmentation
        amount_quantiles = pd.qcut(X['Amount'], q=3, labels=['low', 'medium', 'high'])
        
        for segment in ['low', 'medium', 'high']:
            mask = amount_quantiles == segment
            self.segment_models[f'amount_{segment}'] = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state
            ).fit(X[mask])
    
    def _generate_meta_features(self, X):
        """Generate features for meta-learner"""
        meta_features = []
        
        # Base model scores
        for name, model in self.base_models.items():
            if hasattr(model, 'decision_function'):
                scores = model.decision_function(X)
            else:
                scores = model.score_samples(X)
            meta_features.append(scores)
        
        # Segment model scores
        for name, model in self.segment_models.items():
            scores = model.decision_function(X)
            meta_features.append(scores)
        
        return np.column_stack(meta_features)
    
    def _calculate_feature_weights(self, X, y):
        """Calculate feature importance weights"""
        rf = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=self.random_state
        )
        rf.fit(X, y)
        return rf.feature_importances_
    
    def _get_base_predictions(self, X):
        """Get predictions from base models"""
        predictions = []
        for model in self.base_models.values():
            if hasattr(model, 'decision_function'):
                scores = model.decision_function(X)
            else:
                scores = model.score_samples(X)
            predictions.append(scores)
        return np.column_stack(predictions)
    
    def _get_segment_predictions(self, X):
        """Get predictions from segment-specific models"""
        predictions = []
        amount_quantiles = pd.qcut(X['Amount'], q=3, labels=['low', 'medium', 'high'])
        
        for segment in ['low', 'medium', 'high']:
            mask = amount_quantiles == segment
            model = self.segment_models[f'amount_{segment}']
            scores = np.zeros(len(X))
            scores[mask] = model.decision_function(X[mask])
            predictions.append(scores)
        
        return np.column_stack(predictions)
    
    def _get_current_weights(self):
        """Get current model weights based on time-adaptive performance"""
        weights = np.array(list(self.time_weights.values()))
        return weights / weights.sum()
    
    def _get_weighted_scores(self, X):
        """Calculate weighted anomaly scores"""
        base_scores = self._get_base_predictions(X)
        segment_scores = self._get_segment_predictions(X)
        
        all_scores = np.column_stack([base_scores, segment_scores])
        weights = self._get_current_weights()
        
        return np.average(all_scores, weights=weights, axis=1)
    
    def update_weights(self, performance_metrics):
        """Update time-adaptive weights based on recent performance"""
        self.performance_history.append(performance_metrics)
        
        # Update weights based on recent performance
        if len(self.performance_history) > 1:
            for model_name in self.base_models.keys():
                current_perf = performance_metrics[model_name]
                self.time_weights[model_name] *= (1 + current_perf)
    
    def save_model(self, path):
        """Save ensemble model"""
        model_data = {
            'base_models': self.base_models,
            'segment_models': self.segment_models,
            'meta_learner': self.meta_learner,
            'feature_weights': self.feature_weights,
            'time_weights': dict(self.time_weights),
            'contamination': self.contamination
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load_model(cls, path):
        """Load ensemble model"""
        model_data = joblib.load(path)
        ensemble = cls(contamination=model_data['contamination'])
        ensemble.base_models = model_data['base_models']
        ensemble.segment_models = model_data['segment_models']
        ensemble.meta_learner = model_data['meta_learner']
        ensemble.feature_weights = model_data['feature_weights']
        ensemble.time_weights = defaultdict(lambda: 1.0, model_data['time_weights'])
        return ensemble

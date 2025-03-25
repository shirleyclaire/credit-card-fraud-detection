from sklearn.metrics import (precision_recall_curve, average_precision_score,
                           roc_auc_score, confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        
    def evaluate_model(self, y_true, y_pred, y_scores):
        """Comprehensive model evaluation"""
        # Calculate AUPRC and baseline
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)
        baseline = np.sum(y_true) / len(y_true)  # Random classifier baseline
        
        # Visualize Precision-Recall curve
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, label=f'AUPRC = {auprc:.3f}')
        plt.axhline(y=baseline, color='r', linestyle='--', 
                   label=f'Baseline = {baseline:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Calculate metrics at different threshold levels
        metrics_df = self._calculate_threshold_metrics(y_true, y_scores)
        self._plot_threshold_metrics(metrics_df)
        
        # Analyze errors
        self._analyze_errors(y_true, y_pred, self.X)
        
        return {
            'auprc': auprc,
            'baseline': baseline,
            'classification_report': classification_report(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    def cross_validate(self, n_splits=5):
        """Perform stratified cross-validation"""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = {
            'auprc': [], 'precision': [], 'recall': [], 'f1': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            self.model.fit(X_train)
            y_scores = self.model.decision_function(X_val)
            y_pred = self.model.predict(X_val)
            
            cv_scores['auprc'].append(average_precision_score(y_val, y_scores))
            metrics = self._get_metrics_at_fixed_precision(y_val, y_scores)
            cv_scores['precision'].append(metrics['precision'])
            cv_scores['recall'].append(metrics['recall'])
            cv_scores['f1'].append(metrics['f1'])
            
        return pd.DataFrame(cv_scores)
    
    def feature_importance_analysis(self, feature_names):
        """Analyze feature importance using permutation"""
        base_score = average_precision_score(self.y, self.model.decision_function(self.X))
        importance_scores = []
        
        for feature in feature_names:
            X_permuted = self.X.copy()
            X_permuted[feature] = np.random.permutation(X_permuted[feature])
            permuted_score = average_precision_score(
                self.y, 
                self.model.decision_function(X_permuted)
            )
            importance_scores.append(base_score - permuted_score)
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=True)
        
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance (AUPRC Decrease)')
        plt.title('Feature Importance Analysis')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def _calculate_threshold_metrics(self, y_true, y_scores):
        """Calculate metrics at different thresholds"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return pd.DataFrame({
            'threshold': thresholds,
            'precision': precision[:-1],
            'recall': recall[:-1],
            'f1': f1_scores[:-1]
        })
    
    def _plot_threshold_metrics(self, metrics_df):
        """Visualize metrics across thresholds"""
        plt.figure(figsize=(12, 6))
        for metric in ['precision', 'recall', 'f1']:
            plt.plot(metrics_df['threshold'], 
                    metrics_df[metric], 
                    label=metric.capitalize())
        plt.xlabel('Decision Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs Decision Threshold')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def _analyze_errors(self, y_true, y_pred, X):
        """Analyze false positives and false negatives"""
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)
        
        fp_analysis = X[fp_mask].describe()
        fn_analysis = X[fn_mask].describe()
        
        print("\nFalse Positive Analysis:")
        print(fp_analysis)
        print("\nFalse Negative Analysis:")
        print(fn_analysis)
    
    def _get_metrics_at_fixed_precision(self, y_true, y_scores, target_precision=0.95):
        """Calculate metrics at fixed precision level"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        idx = np.argmin(np.abs(precision - target_precision))
        
        return {
            'precision': precision[idx],
            'recall': recall[idx],
            'threshold': thresholds[idx],
            'f1': 2 * (precision[idx] * recall[idx]) / 
                 (precision[idx] + recall[idx] + 1e-8)
        }

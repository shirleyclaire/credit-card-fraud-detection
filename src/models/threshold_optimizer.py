import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from dataclasses import dataclass

@dataclass
class CostConfig:
    false_positive_cost: float = 10.0  # Cost of investigating legitimate transaction
    false_negative_cost: float = 100.0  # Average cost of missed fraud
    max_review_capacity: int = 100  # Maximum daily reviews
    target_precision: float = 0.95  # Minimum required precision

class ThresholdOptimizer:
    def __init__(self, cost_config=None):
        self.cost_config = cost_config or CostConfig()
        self.thresholds_history = []
        self.performance_history = []
    
    def optimize_threshold(self, y_true, scores, transaction_amounts=None):
        """Find optimal threshold based on cost-benefit analysis"""
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        
        # Calculate metrics for each threshold
        results = []
        for threshold in thresholds:
            metrics = self._calculate_metrics(y_true, scores, threshold, transaction_amounts)
            results.append({
                'threshold': threshold,
                **metrics
            })
        
        threshold_df = pd.DataFrame(results)
        
        # Find optimal threshold based on different criteria
        optimal_thresholds = {
            'cost_optimal': self._find_cost_optimal_threshold(threshold_df),
            'capacity_optimal': self._find_capacity_optimal_threshold(threshold_df),
            'precision_optimal': self._find_precision_optimal_threshold(threshold_df)
        }
        
        # Visualize results
        self._plot_threshold_analysis(threshold_df)
        
        # Store history for adaptive thresholding
        self.thresholds_history.append(optimal_thresholds['cost_optimal'])
        self.performance_history.append(threshold_df)
        
        return optimal_thresholds
    
    def _calculate_metrics(self, y_true, scores, threshold, transaction_amounts=None):
        """Calculate comprehensive metrics for a threshold"""
        predictions = (scores >= threshold).astype(int)
        
        tp = np.sum((predictions == 1) & (y_true == 1))
        fp = np.sum((predictions == 1) & (y_true == 0))
        fn = np.sum((predictions == 0) & (y_true == 1))
        
        # Calculate costs
        review_cost = fp * self.cost_config.false_positive_cost
        missed_fraud_cost = fn * self.cost_config.false_negative_cost
        if transaction_amounts is not None:
            missed_fraud_cost = np.sum(
                transaction_amounts[(predictions == 0) & (y_true == 1)]
            )
        
        total_cost = review_cost + missed_fraud_cost
        
        return {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'review_cost': review_cost,
            'missed_fraud_cost': missed_fraud_cost,
            'total_cost': total_cost,
            'daily_reviews': (fp + tp) / 30  # Assuming monthly data
        }
    
    def _find_cost_optimal_threshold(self, threshold_df):
        """Find threshold that minimizes total cost"""
        return threshold_df.loc[threshold_df['total_cost'].idxmin()]['threshold']
    
    def _find_capacity_optimal_threshold(self, threshold_df):
        """Find optimal threshold considering review capacity"""
        capacity_df = threshold_df[
            threshold_df['daily_reviews'] <= self.cost_config.max_review_capacity
        ]
        return capacity_df.loc[capacity_df['recall'].idxmax()]['threshold']
    
    def _find_precision_optimal_threshold(self, threshold_df):
        """Find threshold meeting minimum precision requirement"""
        precision_df = threshold_df[
            threshold_df['precision'] >= self.cost_config.target_precision
        ]
        return precision_df.loc[precision_df['recall'].idxmax()]['threshold']
    
    def _plot_threshold_analysis(self, threshold_df):
        """Visualize threshold analysis results"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot precision-recall tradeoff
        ax1.plot(threshold_df['threshold'], threshold_df['precision'], 
                label='Precision')
        ax1.plot(threshold_df['threshold'], threshold_df['recall'], 
                label='Recall')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Precision-Recall Tradeoff')
        ax1.legend()
        ax1.grid(True)
        
        # Plot costs
        ax2.plot(threshold_df['threshold'], threshold_df['review_cost'], 
                label='Review Cost')
        ax2.plot(threshold_df['threshold'], threshold_df['missed_fraud_cost'], 
                label='Missed Fraud Cost')
        ax2.plot(threshold_df['threshold'], threshold_df['total_cost'], 
                label='Total Cost')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Cost')
        ax2.set_title('Cost Analysis')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_adaptive_threshold(self, window_size=5):
        """Calculate adaptive threshold based on recent performance"""
        if len(self.thresholds_history) < window_size:
            return np.mean(self.thresholds_history)
        
        return np.mean(self.thresholds_history[-window_size:])
    
    def generate_recommendation(self, threshold_metrics):
        """Generate threshold recommendation with explanation"""
        recommendation = {
            'cost_optimal': {
                'threshold': threshold_metrics['cost_optimal'],
                'explanation': 'Minimizes total cost of false positives and false negatives'
            },
            'capacity_optimal': {
                'threshold': threshold_metrics['capacity_optimal'],
                'explanation': f'Optimizes recall within {self.cost_config.max_review_capacity} daily review capacity'
            },
            'precision_optimal': {
                'threshold': threshold_metrics['precision_optimal'],
                'explanation': f'Maximizes recall while maintaining {self.cost_config.target_precision:.0%} precision'
            }
        }
        return recommendation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from scipy import stats
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from concurrent.futures import ThreadPoolExecutor
from .ensemble_detector import EnsembleFraudDetector

class ModelComparison:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'isolation_forest': IsolationForest(
                contamination='auto',
                random_state=random_state,
                n_jobs=-1
            ),
            'one_class_svm': OneClassSVM(
                kernel='rbf',
                nu=0.001  # Approximately contamination rate
            ),
            'local_outlier_factor': LocalOutlierFactor(
                novelty=True,
                contamination='auto',
                n_jobs=-1
            ),
            'random_forest': RandomForestClassifier(
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            )
        }
        self.models.update({
            'ensemble': EnsembleFraudDetector(
                contamination=0.002,
                random_state=random_state
            )
        })
        self.results_ = {}
        
    def fit_evaluate(self, X_train, X_test, y_train, y_test):
        """Fit and evaluate all models"""
        results = []
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Measure training time
            start_time = time()
            if name == 'random_forest':
                model.fit(X_train, y_train)
            else:
                model.fit(X_train)
            training_time = time() - start_time
            
            # Measure inference time
            start_time = time()
            if hasattr(model, 'decision_function'):
                y_scores = model.decision_function(X_test)
            elif hasattr(model, 'score_samples'):
                y_scores = -model.score_samples(X_test)
            else:
                y_scores = model.predict_proba(X_test)[:, 1]
            inference_time = time() - start_time
            
            # Calculate metrics
            auprc = average_precision_score(y_test, y_scores)
            precision, recall, _ = precision_recall_curve(y_test, y_scores)
            
            results.append({
                'model': name,
                'auprc': auprc,
                'training_time': training_time,
                'inference_time': inference_time,
                'precision': precision,
                'recall': recall
            })
        
        self.results_ = pd.DataFrame(results)
        return self.results_
    
    def cross_validate(self, X, y, n_splits=5):
        """Perform cross-validation for all models"""
        cv_results = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                            random_state=self.random_state)
        
        for name, model in self.models.items():
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                if name == 'random_forest':
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train)
                
                if hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_val)
                elif hasattr(model, 'score_samples'):
                    y_scores = -model.score_samples(X_val)
                else:
                    y_scores = model.predict_proba(X_val)[:, 1]
                
                auprc = average_precision_score(y_val, y_scores)
                fold_scores.append(auprc)
            
            cv_results.append({
                'model': name,
                'cv_mean': np.mean(fold_scores),
                'cv_std': np.std(fold_scores)
            })
        
        return pd.DataFrame(cv_results)
    
    def statistical_comparison(self):
        """Perform statistical tests to compare model performance"""
        if not self.results_.empty:
            # Perform Friedman test
            friedman_statistic, friedman_pvalue = stats.friedmanchisquare(
                *[self.results_[self.results_['model'] == model]['auprc']
                  for model in self.models.keys()]
            )
            
            return {
                'friedman_statistic': friedman_statistic,
                'friedman_pvalue': friedman_pvalue
            }
    
    def plot_performance_comparison(self):
        """Visualize performance comparison"""
        if not self.results_.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot AUPRC comparison
            sns.barplot(data=self.results_, x='model', y='auprc', ax=ax1)
            ax1.set_title('AUPRC by Model')
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
            
            # Plot computational efficiency
            efficiency_data = self.results_.melt(
                id_vars=['model'],
                value_vars=['training_time', 'inference_time'],
                var_name='metric',
                value_name='seconds'
            )
            sns.barplot(data=efficiency_data, x='model', y='seconds', 
                       hue='metric', ax=ax2)
            ax2.set_title('Computational Efficiency')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            # Plot PR curves
            plt.figure(figsize=(10, 6))
            for _, row in self.results_.iterrows():
                plt.plot(row['recall'], row['precision'], 
                        label=f"{row['model']} (AUPRC={row['auprc']:.3f})")
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves')
            plt.legend()
            plt.grid(True)
            plt.show()
    
    def generate_recommendation(self):
        """Generate model recommendation based on performance metrics"""
        if not self.results_.empty:
            best_model = self.results_.loc[self.results_['auprc'].idxmax()]
            fastest_model = self.results_.loc[self.results_['inference_time'].idxmin()]
            
            recommendation = {
                'best_overall': {
                    'model': best_model['model'],
                    'auprc': best_model['auprc'],
                    'explanation': 'Best performance in terms of AUPRC'
                },
                'fastest': {
                    'model': fastest_model['model'],
                    'inference_time': fastest_model['inference_time'],
                    'explanation': 'Fastest inference time for real-time detection'
                }
            }
            
            # Add robustness analysis
            cv_results = self.cross_validate(X, y)
            most_stable = cv_results.loc[cv_results['cv_std'].idxmin()]
            recommendation['most_stable'] = {
                'model': most_stable['model'],
                'cv_std': most_stable['cv_std'],
                'explanation': 'Most consistent performance across different data splits'
            }
            
            return recommendation

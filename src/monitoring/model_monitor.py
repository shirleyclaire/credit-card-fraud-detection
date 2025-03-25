import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
import json

@dataclass
class MonitoringConfig:
    drift_threshold: float = 0.05  # p-value threshold for drift detection
    performance_threshold: float = 0.1  # Maximum allowed performance degradation
    window_size: int = 1000  # Number of transactions for rolling metrics
    alert_cooldown: int = 3600  # Minimum seconds between alerts
    retraining_threshold: float = 0.2  # Performance degradation triggering retraining

class ModelMonitor:
    def __init__(self, config=None):
        self.config = config or MonitoringConfig()
        self.reference_data = None
        self.current_window = []
        self.performance_history = []
        self.last_alert_time = None
        self.drift_scores = []
        
    def initialize_reference(self, X_reference, y_reference, scores_reference):
        """Initialize reference distribution"""
        self.reference_data = {
            'features': X_reference,
            'labels': y_reference,
            'scores': scores_reference,
            'distributions': self._calculate_distributions(X_reference),
            'auprc': average_precision_score(y_reference, scores_reference)
        }
        
    def monitor_prediction(self, timestamp, features, prediction, score, actual=None):
        """Monitor single prediction"""
        # Add to current window
        self.current_window.append({
            'timestamp': timestamp,
            'features': features,
            'prediction': prediction,
            'score': score,
            'actual': actual
        })
        
        # Check window size
        if len(self.current_window) >= self.config.window_size:
            self._analyze_window()
        
        # Check for drift
        drift_detected = self._check_for_drift()
        
        # Track performance if labels available
        if actual is not None:
            self._update_performance_metrics(actual, score)
        
        return drift_detected
    
    def _analyze_window(self):
        """Analyze current window of predictions"""
        window_data = pd.DataFrame(self.current_window)
        
        # Calculate drift scores
        drift_score = self._calculate_drift_score(window_data)
        self.drift_scores.append({
            'timestamp': datetime.now(),
            'score': drift_score
        })
        
        # Check performance
        if 'actual' in window_data.columns:
            current_auprc = average_precision_score(
                window_data['actual'], 
                window_data['score']
            )
            
            performance_drop = self.reference_data['auprc'] - current_auprc
            
            if performance_drop > self.config.performance_threshold:
                self._trigger_alert('Performance Degradation', {
                    'current_auprc': current_auprc,
                    'reference_auprc': self.reference_data['auprc'],
                    'drop': performance_drop
                })
        
        # Reset window
        self.current_window = []
    
    def _calculate_drift_score(self, window_data):
        """Calculate drift score using Kolmogorov-Smirnov test"""
        drift_scores = {}
        
        for feature in self.reference_data['distributions'].keys():
            try:
                statistic, p_value = stats.ks_2samp(
                    self.reference_data['distributions'][feature],
                    window_data['features'].apply(lambda x: x[feature])
                )
                drift_scores[feature] = p_value
            except:
                continue
        
        return np.mean(list(drift_scores.values()))
    
    def _check_for_drift(self):
        """Check if drift requires action"""
        if len(self.drift_scores) < 2:
            return False
            
        recent_scores = pd.DataFrame(self.drift_scores[-10:])
        if (recent_scores['score'] < self.config.drift_threshold).any():
            self._trigger_alert('Data Drift Detected', {
                'drift_scores': recent_scores['score'].tolist()
            })
            return True
        return False
    
    def _update_performance_metrics(self, actual, score):
        """Update performance tracking"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'actual': actual,
            'score': score
        })
    
    def _trigger_alert(self, alert_type, details):
        """Trigger monitoring alert"""
        current_time = datetime.now()
        
        # Check alert cooldown
        if (self.last_alert_time is None or 
            (current_time - self.last_alert_time).seconds > self.config.alert_cooldown):
            
            logging.warning(f"Alert: {alert_type}\nDetails: {json.dumps(details, indent=2)}")
            self.last_alert_time = current_time
            
            # Check if retraining is needed
            if (alert_type == 'Performance Degradation' and 
                details['drop'] > self.config.retraining_threshold):
                self._trigger_retraining()
    
    def _trigger_retraining(self):
        """Trigger model retraining"""
        logging.info("Triggering model retraining due to significant performance degradation")
        # Implement retraining logic here
    
    def create_monitoring_dashboard(self):
        """Create interactive monitoring dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Prediction Distribution', 'Drift Scores',
                          'Performance Metrics', 'Feature Importance')
        )
        
        # Prediction distribution
        scores_df = pd.DataFrame(self.current_window)
        fig.add_trace(
            go.Histogram(x=scores_df['score'], name='Current'),
            row=1, col=1
        )
        
        # Drift scores
        drift_df = pd.DataFrame(self.drift_scores)
        fig.add_trace(
            go.Scatter(x=drift_df['timestamp'], y=drift_df['score'],
                      mode='lines', name='Drift Score'),
            row=1, col=2
        )
        
        # Performance metrics
        if self.performance_history:
            perf_df = pd.DataFrame(self.performance_history)
            window_auprc = []
            for i in range(0, len(perf_df), self.config.window_size):
                window = perf_df.iloc[i:i+self.config.window_size]
                auprc = average_precision_score(window['actual'], window['score'])
                window_auprc.append({
                    'timestamp': window['timestamp'].iloc[-1],
                    'auprc': auprc
                })
            
            auprc_df = pd.DataFrame(window_auprc)
            fig.add_trace(
                go.Scatter(x=auprc_df['timestamp'], y=auprc_df['auprc'],
                          mode='lines', name='AUPRC'),
                row=2, col=1
            )
        
        fig.update_layout(height=800, title_text="Model Monitoring Dashboard")
        return fig
    
    def generate_monitoring_report(self):
        """Generate comprehensive monitoring report"""
        report = {
            'timestamp': datetime.now(),
            'drift_status': {
                'current_score': self.drift_scores[-1]['score'] if self.drift_scores else None,
                'trend': self._calculate_drift_trend()
            },
            'performance_metrics': self._calculate_performance_metrics(),
            'alerts': self._get_recent_alerts(),
            'retraining_recommendation': self._get_retraining_recommendation()
        }
        return report

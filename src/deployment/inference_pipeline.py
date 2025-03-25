import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import joblib
from collections import deque
from functools import lru_cache
import logging
from ..monitoring.model_monitor import ModelMonitor, MonitoringConfig

class RealTimeInference:
    def __init__(self, model_path, preprocessor_path, cache_size=10000):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.cache_size = cache_size
        self.card_history = {}
        self.performance_log = deque(maxlen=1000)
        
        # Configure logging
        logging.basicConfig(
            filename='fraud_detection.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize executor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize model monitor
        self.monitor = ModelMonitor(MonitoringConfig())
    
    @lru_cache(maxsize=1000)
    def _compute_card_features(self, card_id, timestamp):
        """Compute cached card-level features"""
        history = self.card_history.get(card_id, [])
        if not history:
            return np.zeros(5)  # Default features for new cards
            
        recent_txns = [tx for tx in history 
                      if 0 <= timestamp - tx['timestamp'] <= 86400]  # Last 24h
        
        return np.array([
            len(recent_txns),  # Transaction count
            np.mean([tx['amount'] for tx in recent_txns]) if recent_txns else 0,
            np.std([tx['amount'] for tx in recent_txns]) if recent_txns else 0,
            time.time() - history[-1]['timestamp'],  # Time since last transaction
            len(set(tx['merchant'] for tx in recent_txns))  # Unique merchants
        ])
    
    def _preprocess_transaction(self, transaction):
        """Preprocess single transaction for inference"""
        try:
            # Extract basic features
            features = {
                'Amount': transaction['amount'],
                'Time': transaction['timestamp']
            }
            
            # Add card-level features
            card_features = self._compute_card_features(
                transaction['card_id'], 
                transaction['timestamp']
            )
            
            # Combine features
            for i, val in enumerate(card_features):
                features[f'card_feature_{i}'] = val
            
            # Apply standard preprocessing
            return self.preprocessor.transform(features)
            
        except Exception as e:
            logging.error(f"Preprocessing error: {str(e)}")
            raise
    
    def initialize_monitoring(self, X_reference, y_reference):
        """Initialize monitoring with reference data"""
        reference_scores = self.model.decision_function(X_reference)
        self.monitor.initialize_reference(X_reference, y_reference, reference_scores)
    
    def predict_transaction(self, transaction):
        """Real-time prediction with latency tracking"""
        start_time = time.time()
        try:
            # Preprocess
            features = self._preprocess_transaction(transaction)
            
            # Generate prediction
            score = self.model.decision_function(features)
            prediction = score >= self.model.threshold
            
            # Update card history
            self._update_card_history(transaction)
            
            # Log performance
            latency = time.time() - start_time
            self.performance_log.append(latency)
            
            logging.info(
                f"Transaction processed - Card: {transaction['card_id']}, "
                f"Amount: {transaction['amount']}, Score: {score:.3f}, "
                f"Latency: {latency*1000:.2f}ms"
            )
            
            # Add monitoring
            drift_detected = self.monitor.monitor_prediction(
                timestamp=transaction['timestamp'],
                features=features,
                prediction=prediction,
                score=score,
                actual=transaction.get('label')  # If available
            )
            
            if drift_detected:
                logging.warning("Drift detected in transaction patterns")
            
            return {
                'fraud_probability': float(score),
                'is_fraud': bool(prediction),
                'latency_ms': latency * 1000,
                'drift_detected': drift_detected
            }
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return {
                'error': str(e),
                'is_fraud': True  # Fail safe: flag for review
            }
    
    def _update_card_history(self, transaction):
        """Update card transaction history"""
        card_id = transaction['card_id']
        if card_id not in self.card_history:
            self.card_history[card_id] = []
        
        self.card_history[card_id].append({
            'timestamp': transaction['timestamp'],
            'amount': transaction['amount'],
            'merchant': transaction['merchant']
        })
        
        # Maintain cache size
        if len(self.card_history[card_id]) > self.cache_size:
            self.card_history[card_id].pop(0)
    
    def get_performance_stats(self):
        """Calculate performance statistics"""
        latencies = np.array(self.performance_log)
        return {
            'mean_latency_ms': np.mean(latencies) * 1000,
            'p95_latency_ms': np.percentile(latencies, 95) * 1000,
            'p99_latency_ms': np.percentile(latencies, 99) * 1000,
            'transactions_processed': len(self.performance_log)
        }

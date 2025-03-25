import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Dict
import json
import heapq

@dataclass
class RiskFactor:
    name: str
    score: float
    description: str
    importance: float

class InvestigationSystem:
    def __init__(self, model, preprocessor, history_window_days=30):
        self.model = model
        self.preprocessor = preprocessor
        self.history_window_days = history_window_days
        self.case_history = []
        self.feedback_data = []
        
    def generate_case_report(self, transaction, card_history):
        """Generate comprehensive investigation report"""
        # Get model prediction and risk factors
        risk_factors = self._analyze_risk_factors(transaction, card_history)
        total_risk_score = sum(rf.score * rf.importance for rf in risk_factors)
        
        # Generate visualizations
        transaction_context = self._generate_context_visualization(
            transaction, card_history
        )
        
        # Find similar cases
        similar_cases = self._find_similar_cases(transaction, risk_factors)
        
        return {
            'transaction_details': transaction,
            'risk_score': total_risk_score,
            'risk_factors': risk_factors,
            'transaction_context': transaction_context,
            'similar_cases': similar_cases,
            'investigation_priority': self._calculate_priority(
                total_risk_score, transaction['amount']
            )
        }
    
    def _analyze_risk_factors(self, transaction, card_history):
        """Analyze contributing risk factors"""
        risk_factors = []
        
        # Analyze amount patterns
        avg_amount = np.mean([tx['amount'] for tx in card_history])
        if transaction['amount'] > avg_amount * 3:
            risk_factors.append(RiskFactor(
                name="Unusual Amount",
                score=0.8,
                description=f"Amount is {transaction['amount']/avg_amount:.1f}x higher than average",
                importance=0.3
            ))
        
        # Analyze time patterns
        if self._is_unusual_time(transaction['timestamp']):
            risk_factors.append(RiskFactor(
                name="Unusual Time",
                score=0.6,
                description="Transaction occurred during unusual hours",
                importance=0.2
            ))
        
        # Analyze location/merchant patterns
        if self._is_unusual_merchant(transaction, card_history):
            risk_factors.append(RiskFactor(
                name="Unusual Merchant",
                score=0.7,
                description="First time transaction with this merchant",
                importance=0.25
            ))
        
        # Add model-based anomaly factors
        model_factors = self._get_model_risk_factors(transaction)
        risk_factors.extend(model_factors)
        
        return risk_factors
    
    def _generate_context_visualization(self, transaction, card_history):
        """Generate context visualizations"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Transaction Amount History', 'Time Pattern',
                          'Merchant Distribution', 'Risk Factor Breakdown')
        )
        
        # Amount history
        amounts = [tx['amount'] for tx in card_history]
        timestamps = [tx['timestamp'] for tx in card_history]
        fig.add_trace(
            go.Scatter(x=timestamps, y=amounts, mode='lines+markers',
                      name='Previous Transactions'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[transaction['timestamp']], y=[transaction['amount']],
                      mode='markers', name='Current Transaction',
                      marker=dict(size=12, color='red')),
            row=1, col=1
        )
        
        # Time pattern
        hour_dist = self._calculate_hour_distribution(card_history)
        fig.add_trace(
            go.Bar(x=list(hour_dist.keys()), y=list(hour_dist.values()),
                  name='Hour Distribution'),
            row=1, col=2
        )
        
        # Merchant distribution
        merchant_dist = self._calculate_merchant_distribution(card_history)
        fig.add_trace(
            go.Pie(labels=list(merchant_dist.keys()),
                   values=list(merchant_dist.values())),
            row=2, col=1
        )
        
        return fig
    
    def _find_similar_cases(self, transaction, risk_factors, n_cases=5):
        """Find similar historical cases"""
        similar_cases = []
        for case in self.case_history:
            similarity_score = self._calculate_case_similarity(
                transaction, risk_factors, case
            )
            heapq.heappush(similar_cases, 
                          (-similarity_score, case))  # Negative for max-heap
            if len(similar_cases) > n_cases:
                heapq.heappop(similar_cases)
        
        return [case for _, case in sorted(similar_cases)]
    
    def add_investigator_feedback(self, case_id, feedback):
        """Add investigator feedback for continuous improvement"""
        self.feedback_data.append({
            'case_id': case_id,
            'timestamp': datetime.now(),
            'feedback': feedback
        })
        
        # Update case history
        self._update_case_history(case_id, feedback)
        
        # Trigger model update if needed
        if len(self.feedback_data) >= 100:  # Batch size
            self._update_model_with_feedback()
    
    def get_investigation_queue(self, max_cases=100):
        """Get prioritized investigation queue"""
        pending_cases = self._get_pending_cases()
        
        # Score and sort cases
        scored_cases = []
        for case in pending_cases:
            priority_score = self._calculate_priority(
                case['risk_score'],
                case['transaction']['amount']
            )
            heapq.heappush(scored_cases, (-priority_score, case))
        
        # Return top cases
        return [case for _, case in heapq.nlargest(max_cases, scored_cases)]
    
    def _calculate_priority(self, risk_score, amount):
        """Calculate investigation priority score"""
        # Combine risk score with amount-based importance
        amount_factor = np.log1p(amount) / 10  # Dampened amount influence
        time_factor = 1.0  # Could be adjusted based on age of case
        return risk_score * (0.7 + 0.3 * amount_factor) * time_factor
    
    def _is_unusual_time(self, timestamp):
        """Check if transaction time is unusual"""
        hour = datetime.fromtimestamp(timestamp).hour
        return hour >= 0 and hour <= 5  # Unusual hours (midnight to 5am)
    
    def _is_unusual_merchant(self, transaction, card_history):
        """Check if merchant is unusual for this card"""
        known_merchants = set(tx['merchant'] for tx in card_history)
        return transaction['merchant'] not in known_merchants
    
    def _get_model_risk_factors(self, transaction):
        """Extract risk factors from model decision"""
        # Get feature importance for this prediction
        features = self.preprocessor.transform(pd.DataFrame([transaction]))
        importance = self.model.feature_importances(features)
        
        return [
            RiskFactor(
                name=f"Model Factor: {feature}",
                score=score,
                description=f"Model identified unusual pattern in {feature}",
                importance=0.25/len(importance)
            )
            for feature, score in importance.items()
            if score > 0.1  # Only include significant factors
        ]

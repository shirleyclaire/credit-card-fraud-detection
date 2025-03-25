import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import dash
from dash import dcc, html  # Updated imports
from dash.dependencies import Input, Output
import plotly.express as px

class ModelInterpreter:
    def __init__(self, model, X_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model.model)
        self.prototypes = None
        
    def explain_prediction(self, transaction):
        """Generate SHAP explanation for a single transaction"""
        shap_values = self.explainer.shap_values(transaction)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, transaction, feature_names=self.feature_names,
                         show=False)
        plt.title('Feature Contributions to Fraud Score')
        plt.tight_layout()
        plt.show()
        
        return pd.Series(shap_values[0], index=self.feature_names)
    
    def extract_decision_rules(self, max_depth=3):
        """Extract interpretable rules using a surrogate model"""
        # Train surrogate decision tree
        surrogate = DecisionTreeClassifier(max_depth=max_depth)
        predictions = self.model.predict(self.X_train)
        surrogate.fit(self.X_train, predictions)
        
        # Extract rules
        rules = self._get_decision_rules(surrogate)
        return pd.DataFrame(rules, columns=['rule', 'support', 'confidence'])
    
    def generate_prototypes(self, n_clusters=5):
        """Generate fraud prototypes using clustering"""
        # Get fraud predictions
        predictions = self.model.predict(self.X_train)
        fraud_samples = self.X_train[predictions == 1]
        
        # Cluster fraud cases
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(fraud_samples)
        
        # Calculate prototype characteristics
        self.prototypes = []
        for i in range(n_clusters):
            cluster_data = fraud_samples[clusters == i]
            prototype = {
                'centroid': cluster_data.mean(),
                'std': cluster_data.std(),
                'size': len(cluster_data)
            }
            self.prototypes.append(prototype)
        
        return self.prototypes
    
    def explain_with_prototypes(self, transaction):
        """Find similar fraud prototypes"""
        if self.prototypes is None:
            self.generate_prototypes()
            
        similarities = []
        for i, prototype in enumerate(self.prototypes):
            # Calculate Mahalanobis distance
            diff = transaction - prototype['centroid']
            std = prototype['std'].replace(0, 1)  # Avoid division by zero
            distance = np.sqrt(np.sum((diff / std) ** 2))
            
            similarities.append({
                'prototype': i,
                'distance': distance,
                'size': prototype['size']
            })
            
        return pd.DataFrame(similarities).sort_values('distance')
    
    def visualize_decision_path(self, transaction):
        """Visualize decision path in Isolation Forest"""
        # Get decision path
        decision_path = self.model.model.decision_path([transaction])[0]
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        self._plot_decision_path(decision_path.indices)
        plt.title('Decision Path in Isolation Forest')
        plt.show()
    
    def create_dashboard(self):
        """Create interactive dashboard for model interpretation"""
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1('Fraud Detection Model Interpretation'),
            
            dcc.Dropdown(
                id='transaction-selector',
                options=[{'label': f'Transaction {i}', 'value': i} 
                        for i in range(len(self.X_train))],
                value=0
            ),
            
            dcc.Graph(id='feature-importance-plot'),
            dcc.Graph(id='prototype-similarity-plot'),
            
            html.Div(id='decision-rules')
        ])
        
        @app.callback(
            [Output('feature-importance-plot', 'figure'),
             Output('prototype-similarity-plot', 'figure'),
             Output('decision-rules', 'children')],
            [Input('transaction-selector', 'value')]
        )
        def update_dashboard(transaction_idx):
            transaction = self.X_train.iloc[transaction_idx]
            
            # Feature importance
            importance = self.explain_prediction(transaction)
            imp_fig = px.bar(importance)
            
            # Prototype similarity
            similarities = self.explain_with_prototypes(transaction)
            sim_fig = px.bar(similarities, x='prototype', y='distance')
            
            # Decision rules
            rules = self.extract_decision_rules()
            rules_text = [html.P(rule) for rule in rules['rule']]
            
            return imp_fig, sim_fig, rules_text
        
        return app
    
    def _get_decision_rules(self, tree):
        """Extract rules from decision tree"""
        rules = []
        
        def recurse(node, depth, path):
            if tree.tree_.children_left[node] == -1:  # Leaf
                pred = tree.tree_.value[node].argmax()
                rules.append({
                    'rule': ' AND '.join(path),
                    'support': tree.tree_.n_node_samples[node],
                    'confidence': tree.tree_.value[node].max() / 
                                tree.tree_.n_node_samples[node]
                })
                return
                
            feature = self.feature_names[tree.tree_.feature[node]]
            threshold = tree.tree_.threshold[node]
            
            # Left path
            left_path = path + [f"{feature} <= {threshold:.2f}"]
            recurse(tree.tree_.children_left[node], depth + 1, left_path)
            
            # Right path
            right_path = path + [f"{feature} > {threshold:.2f}"]
            recurse(tree.tree_.children_right[node], depth + 1, right_path)
            
        recurse(0, 1, [])
        return rules
    
    def _plot_decision_path(self, path):
        """Helper function to visualize decision path"""
        path_length = len(path)
        x = np.arange(path_length)
        plt.plot(x, path, 'bo-')
        plt.xlabel('Step in Path')
        plt.ylabel('Tree Node Index')
        plt.grid(True)

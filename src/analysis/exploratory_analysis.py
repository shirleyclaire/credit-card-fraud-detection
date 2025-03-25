import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

class ExploratoryAnalysis:
    def __init__(self, df):
        self.df = df
        
    def show_class_distribution(self):
        """Visualize the class imbalance"""
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Class', data=self.df)
        plt.title('Distribution of Fraud vs Normal Transactions')
        fraud_percent = (self.df['Class'].value_counts()[1] / len(self.df)) * 100
        plt.text(0.5, plt.ylim()[1], f'Fraud Cases: {fraud_percent:.2f}%',
                horizontalalignment='center')
        plt.show()
        
    def analyze_feature_distributions(self):
        """Analyze distributions of key features"""
        # Time and Amount distributions
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        sns.histplot(data=self.df, x='Amount', hue='Class', ax=ax1, bins=50)
        ax1.set_title('Distribution of Transaction Amounts')
        ax1.set_yscale('log')
        
        sns.histplot(data=self.df, x='Time', hue='Class', ax=ax2, bins=50)
        ax2.set_title('Distribution of Transaction Time')
        plt.tight_layout()
        plt.show()
        
    def analyze_pca_components(self):
        """Analyze PCA components for fraud patterns"""
        # Select most important V features based on correlation with Class
        correlations = []
        v_features = [col for col in self.df.columns if col.startswith('V')]
        
        for feature in v_features:
            corr = abs(stats.pointbiserialr(self.df[feature], self.df['Class'])[0])
            correlations.append((feature, corr))
        
        # Plot top 6 most correlated features
        top_features = sorted(correlations, key=lambda x: x[1], reverse=True)[:6]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for idx, (feature, _) in enumerate(top_features):
            row = idx // 3
            col = idx % 3
            sns.boxplot(x='Class', y=feature, data=self.df, ax=axes[row, col])
            axes[row, col].set_title(f'{feature} vs Class')
        
        plt.tight_layout()
        plt.show()
        
    def generate_summary_stats(self):
        """Generate statistical summaries"""
        print("\nBasic Statistics for Normal Transactions:")
        print(self.df[self.df['Class'] == 0].describe())
        
        print("\nBasic Statistics for Fraudulent Transactions:")
        print(self.df[self.df['Class'] == 1].describe())
        
    def analyze_time_amount_patterns(self):
        """Analyze relationships between time, amount, and fraud"""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df[self.df['Class'] == 0]['Time'], 
                   self.df[self.df['Class'] == 0]['Amount'],
                   alpha=0.5, label='Normal', s=1)
        plt.scatter(self.df[self.df['Class'] == 1]['Time'], 
                   self.df[self.df['Class'] == 1]['Amount'],
                   color='red', alpha=0.7, label='Fraud', s=10)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amount')
        plt.yscale('log')
        plt.title('Transaction Amount vs Time')
        plt.legend()
        plt.show()

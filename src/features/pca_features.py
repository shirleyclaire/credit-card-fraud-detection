import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns

class PCAFeatureGenerator:
    def __init__(self, n_clusters=3, n_neighbors=20):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        
    def generate_features(self, df):
        """Generate PCA-based anomaly features"""
        # Extract V features
        v_features = [col for col in df.columns if col.startswith('V')]
        X_v = df[v_features].copy()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_v)
        
        # Fit PCA and transform
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Calculate reconstruction error
        X_reconstructed = self.pca.inverse_transform(X_pca)
        reconstruction_error = np.square(X_scaled - X_reconstructed).mean(axis=1)
        
        # Cluster-based features
        self.kmeans.fit(X_pca)
        cluster_distances = self.kmeans.transform(X_pca)
        
        # LOF scores
        self.lof.fit(X_pca)
        lof_scores = -self.lof.score_samples(X_pca)
        
        # Create feature DataFrame
        features = pd.DataFrame()
        
        # Add reconstruction error
        features['pca_reconstruction_error'] = reconstruction_error
        
        # Add cluster distances
        for i in range(self.n_clusters):
            features[f'cluster_{i}_distance'] = cluster_distances[:, i]
        
        # Add LOF scores
        features['local_outlier_factor'] = lof_scores
        
        # Add interaction features for top components
        for i in range(min(3, X_pca.shape[1])):
            for j in range(i+1, min(4, X_pca.shape[1])):
                features[f'pca_{i}_{j}_interaction'] = X_pca[:, i] * X_pca[:, j]
        
        # Add composite score
        features['pca_composite_score'] = (
            (reconstruction_error / reconstruction_error.std()) +
            (lof_scores / lof_scores.std()) +
            (cluster_distances.min(axis=1) / cluster_distances.min(axis=1).std())
        )
        
        return features
    
    def visualize_components(self, df):
        """Visualize PCA components distribution"""
        v_features = [col for col in df.columns if col.startswith('V')]
        X_v = df[v_features].copy()
        X_scaled = self.scaler.fit_transform(X_v)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Plot top 2 components
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[df['Class']==0, 0], X_pca[df['Class']==0, 1], 
                   alpha=0.5, label='Normal', s=1)
        plt.scatter(X_pca[df['Class']==1, 0], X_pca[df['Class']==1, 1], 
                   alpha=0.7, label='Fraud', color='red', s=10)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA Components Distribution')
        plt.legend()
        plt.show()

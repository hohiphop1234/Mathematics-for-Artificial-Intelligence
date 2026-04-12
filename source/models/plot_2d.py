import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data'))
EXPERIMENTS_FOLDER = os.path.abspath(os.path.join(BASE_DIR, '..', 'experiments', 'results'))
os.makedirs(EXPERIMENTS_FOLDER, exist_ok=True)

def visualize_lsa(X, y, title, filename):
    # Reduce to 2 dimensions using Truncated SVD
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X)

    plt.figure(figsize=(10, 7))
    
    # Plot Ham (label 0)
    plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], color='blue', alpha=0.5, label='Ham', s=10)
    # Plot Spam (label 1)
    plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], color='red', alpha=0.5, label='Spam', s=10)

    plt.title(title)
    plt.xlabel('SVD Component 1')
    plt.ylabel('SVD Component 2')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(EXPERIMENTS_FOLDER, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to: {save_path}")

if __name__ == "__main__":
    data_path = os.path.join(DATA_FOLDER, 'spam_features.npz')
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
    else:
        data = np.load(data_path)
        X_train, y_train = data['X_train'], data['y_train']
        
        print("Performing Truncated SVD and generating plot...")
        visualize_lsa(X_train, y_train, "Spam vs Ham Visualization (LSA)", "lsa_visualization.png")
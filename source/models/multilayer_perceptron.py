import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import StandardScaler 

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'spam_features.npz')

class MultilayerPerceptron:
    def __init__(self, input_dim, hidden_dim, k_layers, lr=0.001): 
        self.lr = lr
        self.k_layers = k_layers
        self.weights = []
        self.biases = []
        
        dims = [input_dim] + [hidden_dim] * k_layers + [1]
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros((1, dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -20, 20))) 

    def forward(self, X):
        self.activations = [X]
        self.zs = []
        curr_input = X
        for i in range(self.k_layers):
            z = np.dot(curr_input, self.weights[i]) + self.biases[i]
            curr_input = self.relu(z)
            self.zs.append(z)
            self.activations.append(curr_input)
            
        z_out = np.dot(curr_input, self.weights[-1]) + self.biases[-1]
        output = self.sigmoid(z_out)
        self.zs.append(z_out)
        self.activations.append(output)
        return output

    def backward(self, y):
        m = y.shape[0]
        y = y.reshape(-1, 1)
        delta = self.activations[-1] - y
        
        for i in reversed(range(len(self.weights))):
            dw = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.zs[i-1])
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db

    def fit(self, X, y, epochs=500): 
        for epoch in range(epochs):
            self.forward(X)
            self.backward(y)

    def predict(self, X):
        y_pred = self.forward(X)
        return (y_pred >= 0.5).astype(int).flatten()

def run_experiments():
    if not os.path.exists(DATA_PATH):
        print(f"Lỗi: Không tìm thấy file tại {DATA_PATH}")
        return

    data = np.load(DATA_PATH)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    k_range = list(range(1, 11))
    print(f"{'K':<3} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
    print("-" * 50)

    for k in k_range:
        mlp = MultilayerPerceptron(input_dim=X_train.shape[1], hidden_dim=512, k_layers=k, lr=0.01)
        mlp.fit(X_train, y_train, epochs=300)
        
        y_pred = mlp.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        
        print(f"{k:<3} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f}")

if __name__ == "__main__":
    run_experiments()
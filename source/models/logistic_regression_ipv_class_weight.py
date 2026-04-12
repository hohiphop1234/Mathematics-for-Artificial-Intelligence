import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
EXPERIMENTS_FOLDER = os.path.join(BASE_DIR, "..", "experiments", "results")
os.makedirs(EXPERIMENTS_FOLDER, exist_ok=True)

class ClassWeightLogisticRegression:
    def __init__(self, learning_rate=0.1, iterations=1000, alpha=0.5, beta=0.5):
        self.lr = learning_rate
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.weights = None
        self.bias = None
        self.history = {
            'train_loss': [], 'train_acc': [],
            'test_loss': [], 'test_acc': []
        }

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def compute_loss(self, y, y_predicted, weights=None):
        """
        Luca's Fix: Thêm tham số weights để tính toán Weighted Loss chính xác.
        """
        y = y.reshape(-1, 1)
        y_predicted = y_predicted.reshape(-1, 1)
        y_predicted_clipped = np.clip(y_predicted, 1e-15, 1 - 1e-15)
        
        loss_components = y * np.log(y_predicted_clipped) + (1 - y) * np.log(1 - y_predicted_clipped)
        
        if weights is not None:
            return -np.mean(weights * loss_components)
        return -np.mean(loss_components)

    def compute_accuracy(self, y, y_predicted):
        predictions = (y_predicted.flatten() >= 0.5).astype(int)
        return np.mean(predictions == y.flatten())

    def fit(self, X_train, y_train, X_test, y_test):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_train_vec = y_train.reshape(-1, 1)
        train_weights = (y_train_vec * self.alpha + (1 - y_train_vec) * self.beta)
        
        test_weights = (y_test.reshape(-1, 1) * self.alpha + (1 - y_test.reshape(-1, 1)) * self.beta)

        for i in range(self.iterations):
            # Forward Pass
            z_train = np.dot(X_train, self.weights) + self.bias
            y_pred_train = self.sigmoid(z_train).reshape(-1, 1)
            
            z_test = np.dot(X_test, self.weights) + self.bias
            y_pred_test = self.sigmoid(z_test).reshape(-1, 1)

            train_loss = self.compute_loss(y_train, y_pred_train, train_weights)
            train_acc = self.compute_accuracy(y_train, y_pred_train)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            test_loss = self.compute_loss(y_test, y_pred_test, test_weights)
            test_acc = self.compute_accuracy(y_test, y_pred_test)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            
            error = (y_pred_train - y_train_vec) * train_weights
            dw = (1 / n_samples) * np.dot(X_train.T, error).flatten()
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 100 == 0:
                print(f"Iter {i:4d} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")


    def predict(self, X, threshold=0.7):
        z = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(z)
        return (y_predicted >= threshold).astype(int)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        print(f"\n--- Model Evaluation Results ---")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }

    def save_history(self):
        iters = range(len(self.history['train_loss']))
        plt.figure(figsize=(14, 6))

        # Biểu đồ Loss
        plt.subplot(1, 2, 1)
        plt.plot(iters, self.history['train_loss'], 'r-', label='Train Loss')
        plt.plot(iters, self.history['test_loss'], 'r--', label='Test Loss')
        plt.title('Weighted Loss Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Biểu đồ Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(iters, self.history['train_acc'], 'b-', label='Train Acc')
        plt.plot(iters, self.history['test_acc'], 'b--', label='Test Acc')
        plt.title('Accuracy Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        save_path = os.path.join(EXPERIMENTS_FOLDER, 'class_weight_logistic_regression_performance.png')
        plt.savefig(save_path)
        plt.close()
        print(f"\n[Luca]: Plot saved to: {save_path}")

if __name__ == "__main__":
    DATA_FOLDER_REAL = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data'))
    data_path = os.path.join(DATA_FOLDER_REAL, 'spam_features.npz')
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
    else:
        data = np.load(data_path)
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']

        model = ClassWeightLogisticRegression(learning_rate=1, iterations=10000, alpha=0.5, beta=0.5)
        
        model.fit(X_train, y_train, X_test, y_test)

        model.evaluate(X_test, y_test)
        model.save_history()
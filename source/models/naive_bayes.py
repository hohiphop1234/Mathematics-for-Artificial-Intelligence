import numpy as np
import os
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class SklearnModelWrapper:
    def __init__(self, model):
        self.model = model
        self.name = model.__class__.__name__

    def train(self, X_train, y_train):
        print(f"Training {self.name}...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print(f"\n--- Result for {self.name} ---")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }

    def predict(self, X):
        return self.model.predict(X)
    

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_FOLDER = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data'))
    data_path = os.path.join(DATA_FOLDER, 'spam_features.npz')

    if not os.path.exists(data_path):
        print("Data not found!")
    else:
        data = np.load(data_path)
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']

        models_to_test = [
            SklearnModelWrapper(MultinomialNB(alpha=1.0)),
            SklearnModelWrapper(GaussianNB())
        ]

        results = {}
        for model_wrapper in models_to_test:
            model_wrapper.train(X_train, y_train)
            results[model_wrapper.name] = model_wrapper.evaluate(X_test, y_test)

        print("\n=== FINAL COMPARISON (Accuracy) ===")
        for name, res in results.items():
            print(f"{name:20}: {res['accuracy']:.4f}")
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data'))

if __name__ == "__main__":
    data_path = os.path.join(DATA_FOLDER, 'spam_features.npz')
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
    else:
        data = np.load(data_path)
        X_train, y_train = data['X_train'].astype(np.float64), data['y_train']
        X_test, y_test = data['X_test'].astype(np.float64), data['y_test']

        model = LogisticRegression(solver='liblinear', max_iter=10000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
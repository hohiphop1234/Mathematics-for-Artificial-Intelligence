import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data'))

if __name__ == "__main__":
    data = np.load(os.path.join(DATA_FOLDER, 'spam_features.npz'))
    X_train, y_train = data['X_train'], data['y_train']
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    X_test, y_test = data['X_test'], data['y_test']
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print(type(X_train), type(y_train))
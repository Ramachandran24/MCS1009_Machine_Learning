import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

def load_dataset():
    mat = scipy.io.loadmat("PersonGaitDataSet.mat")
    return {k: v for k, v in mat.items() if not k.startswith('_')}

if __name__ == "__main__":
    # Load and preprocess dataset
    data = load_dataset()
    X = data['X']
    X[np.isnan(X)] = 0  # Replace NaNs with 0
    y = data['Y'].ravel()  # Flatten if y is 2D

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1
    )

    # Train Gaussian Naive Bayes model
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = gnb.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    print(f"Gaussian Naive Bayes model accuracy (in %): {accuracy:.2f}")

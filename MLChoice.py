# Mohammed Ahmed Zakiuddin
# 1001675091

# Instructions:
# 1. Run the program in the command line using the following command: python MLChoice.py <model> <data>
# 2. The program will run and output the results to a file called "output.txt"
# 3. The program will also print the results to the command line
# 4. Example: python MLChoice.py knn Sonar.txt
# 5. Example: python MLChoice.py svm BankNote.txt

import sys
import numpy as np
import pandas as pd

from math import sqrt
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class SVM:
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        # Initialize hyperparameters
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    # Fit method - trains the model
    def fit(self, X, y):
        n_samples, n_features = X.shape # Get the number of samples and features
        y_ = np.where(y <= 0, -1, 1) # Convert labels to -1 and 1

        self.w = np.zeros(n_features) # Initialize weights
        self.b = 0 # Initialize bias

        # Gradient descent
        for _ in range(self.n_iters):   # Iterate over the number of iterations
            for idx, x_i in enumerate(X): # Iterate over the samples
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1 # Check if the condition is true or false
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w) # Update the weights
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])) # Update the weights
                    self.b -= self.lr * y_[idx] # Update the bias

    # Predict method - predicts the output or the labels of test data.
    def predict(self, X):

        linear_output = np.dot(X, self.w) - self.b # w.x - b
        return np.sign(linear_output) # sign(w.x - b)

class k_nearest_neighbors:
    
    def __init__(self, k=3): # Initialize the number of neighbors
        self.k = k

    # Fit method - trains the model
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Predict method - predicts the output or the labels of test data.
    def predict(self, X):
        predictions = []
        for row in X:
            label = self._closest(row)
            predictions.append(label)
        return predictions

    # Find the closest point to the test point
    def _closest(self, row):
        best_dist = float('inf')
        best_index = -1
        for i in range(len(self.X_train)):
            dist = self._distance(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

    # Calculate the distance between two points
    def _distance(self, a, b):
        return sqrt(sum((a - b) ** 2))

class MLChoice:
    def __init__(self, model, data):
        
        self.model = model
        self.data = data
        
        df = pd.read_csv(sys.argv[2])

        dataset = "Sonar" if data == "Sonar.txt" else "BankNote"

        if dataset == "Sonar":
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].map({'R': 0, 'M': 1})
            y = y.values
        
        else:
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Open a file for appending it with the results
        with open('output.txt', 'a') as filename:

            print(f"Dataset: {dataset}\n")
            filename.write(f"Dataset: {dataset}\n")

            print(f"Machine Learning Algorithm Chosen: {model}\n")
            filename.write(f"Machine Learning Algorithm Chosen: {model}\n")

            if model == "knn":
            # Train and test model from scratch
                knn_scratch = k_nearest_neighbors(3)
                knn_scratch.fit(X_train, y_train)
                knn_accuracy = accuracy_score(y_test, knn_scratch.predict(X_test))
                print(f"Accuracy of Training (Scratch): {round(knn_accuracy * 100)}%\n")
                filename.write(f"Accuracy of Training (Scratch): {round(knn_accuracy * 100)}%\n")

                # Train and test model using scikit-learn's KNeighborsClassifier
                scikit_learn = KNeighborsClassifier(3)
                scikit_learn.fit(X_train, y_train)
                scikit_accuracy = accuracy_score(y_test, scikit_learn.predict(X_test))
                print(f"Accuracy of ScikitLearn Function: {round(scikit_accuracy * 100)}%\n")
                filename.write(f"Accuracy of ScikitLearn Function: {round(scikit_accuracy * 100)}%\n")

                # Make predictions for a single point
                X_pred = X_test[0].reshape(1, -1) # # Selects the first row of the test data and reshapes it to a 1D array.
                pred_class = knn_scratch.predict(X_pred)[0]
                print(f"Prediction Point: {X_pred}")
                print(f"Predicted Class: {pred_class}")
                print(f"Actual Class: {y_test[0]}")
                # Write to the file
                filename.write(f"Prediction Point: {X_pred}\n")
                filename.write(f"Predicted Class: {pred_class}\n")
                filename.write(f"Actual Class: {y_test[0]}\n")
                filename.write("\n")

            else:
                # Train and test model from scratch
                svm_scratch = SVM()
                svm_scratch.fit(X_train, y_train)
                svm_accuracy = accuracy_score(y_test, svm_scratch.predict(X_test))
                print(f"Accuracy of Training (Scratch): {round(svm_accuracy * 100)}%\n")
                filename.write(f"Accuracy of Training (Scratch): {round(svm_accuracy * 100)}%\n")

                # Train and test model using scikit-learn's SVC
                scikit_learn = svm.SVC(kernel='rbf', random_state=0)
                scikit_learn.fit(X_train, y_train)
                scikit_accuracy = accuracy_score(y_test, scikit_learn.predict(X_test))
                print(f"Accuracy of ScikitLearn Function: {round(scikit_accuracy * 100)}%\n")
                filename.write(f"Accuracy of ScikitLearn Function: {round(scikit_accuracy * 100)}%\n")

                # Make predictions for a single point
                X_pred = X_test[0].reshape(1, -1) # Selects the first row of the test data and reshapes it to a 1D array.
                pred_class = svm_scratch.predict(X_pred)[0]
                print(f"Prediction Point: {X_pred}")
                print(f"Predicted Class: {pred_class}")
                print(f"Actual Class: {y_test[0]}")
                filename.write(f"Prediction Point: {X_pred}\n")
                filename.write(f"Predicted Class: {pred_class}\n")
                filename.write(f"Actual Class: {y_test[0]}\n")
                filename.write("\n")
        
if __name__ == '__main__':
    
    MLChoice(sys.argv[1], sys.argv[2]) # Pass the model and dataset as command line arguments
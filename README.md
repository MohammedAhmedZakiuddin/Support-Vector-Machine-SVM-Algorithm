# MLChoice - Machine Learning Algorithm Comparison

MLChoice is a Python program that allows you to implement and compare two popular machine learning algorithms, K-Nearest Neighbors (KNN) and Support Vector Machine (SVM), for classification tasks. It provides functionality to train and test these algorithms from scratch and also utilizes scikit-learn for easy comparison.

Introduction: <br />
Machine Learning is a fascinating field with various algorithms to choose from. MLChoice simplifies the process of comparing two popular algorithms, KNN and SVM, using real-world datasets. You can use this tool to understand how different algorithms perform on different datasets.

Features <br />
* Implements KNN and SVM from scratch. <br />
* Utilizes scikit-learn for easy comparison. <br />
* Supports two datasets: Sonar and Banknote. <br />
* Splits the dataset into training and testing data using train_test_split. <br />
* No pre-processing of data required; everything is handled in the code.  <br />

Make sure you have Python installed (version 3.7 or higher) and install the required packages: <br />
pip install -r requirements.txt <br />

Usage: <br />
To use MLChoice, you need to provide two command-line arguments: <br />

1. The machine learning algorithm to use, either knn or svm. <br />
2. The dataset to test, either sonar or banknote. <br />

You can run MLChoice as follows: <br />

python MLChoice.py knn sonar <br />
python MLChoice.py svm banknote <br />

Datasets: <br />
MLChoice supports two datasets for testing: <br />

Sonar Dataset: <br />

* Involves predicting whether an object is a mine (M) or a rock (R) based on sonar return strength at different angles. <br />
* 60 input variables and 1 output variable. <br />

Banknote Dataset:<br />

* Predicts the authenticity of a banknote (0 for authentic, 1 for inauthentic) based on various measures taken from a photograph. <br />
* Features include Variance of Wavelet Transformed image, Skewness, Kurtosis, and Entropy. <br />

Example: <br />
Here's an example of running MLChoice with KNN on the Sonar dataset: <br />

python MLChoice.py knn sonar <br />

Results: 
MLChoice will provide results, including accuracy, precision, recall, and F1-score for both your custom implementation of the algorithm and scikit-learn's implementation. You can analyze these results to understand how different algorithms perform on the selected dataset.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Feel free to use and extend this tool for your machine learning experiments. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request. Happy coding!

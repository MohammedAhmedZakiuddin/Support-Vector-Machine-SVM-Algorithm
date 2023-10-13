# MLChoice - Machine Learning Algorithm Comparison

MLChoice is a Python program that allows you to implement and compare two popular machine learning algorithms, K-Nearest Neighbors (KNN) and Support Vector Machine (SVM), for classification tasks. It provides functionality to train and test these algorithms from scratch and also utilizes scikit-learn for easy comparison.

Introduction
Machine Learning is a fascinating field with various algorithms to choose from. MLChoice simplifies the process of comparing two popular algorithms, KNN and SVM, using real-world datasets. You can use this tool to understand how different algorithms perform on different datasets.

Features
Implements KNN and SVM from scratch.
Utilizes scikit-learn for easy comparison.
Supports two datasets: Sonar and Banknote.
Splits the dataset into training and testing data using train_test_split.
No pre-processing of data required; everything is handled in the code.
Installation
Clone this repository to your local machine:
bash
Copy code
git clone https://github.com/yourusername/MLChoice.git
Change your directory to MLChoice:
bash
Copy code
cd MLChoice
Make sure you have Python installed (version 3.7 or higher) and install the required packages:
bash
Copy code
pip install -r requirements.txt
Usage
To use MLChoice, you need to provide two command-line arguments:

The machine learning algorithm to use, either knn or svm.
The dataset to test, either sonar or banknote.
You can run MLChoice as follows:

bash
Copy code
python MLChoice.py knn sonar
bash
Copy code
python MLChoice.py svm banknote
Datasets
MLChoice supports two datasets for testing:

Sonar Dataset:

Involves predicting whether an object is a mine (M) or a rock (R) based on sonar return strength at different angles.
60 input variables and 1 output variable.
Banknote Dataset:

Predicts the authenticity of a banknote (0 for authentic, 1 for inauthentic) based on various measures taken from a photograph.
Features include Variance of Wavelet Transformed image, Skewness, Kurtosis, and Entropy.
Example
Here's an example of running MLChoice with KNN on the Sonar dataset:

bash
Copy code
python MLChoice.py knn sonar
Results
MLChoice will provide results, including accuracy, precision, recall, and F1-score for both your custom implementation of the algorithm and scikit-learn's implementation. You can analyze these results to understand how different algorithms perform on the selected dataset.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Feel free to use and extend this tool for your machine learning experiments. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request. Happy coding!

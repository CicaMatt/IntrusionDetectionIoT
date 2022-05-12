import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics, naive_bayes
from utils.data_setup import DataSetup
from utils.metrics import Metrics

# Data retrieving
data = DataSetup.data_setup(5)
# print(data.shape)

# Removing all duplicates
data = data.drop_duplicates()

# Shuffling rows of dataframe, done due to consecutive dataset entry being similar to each other
print("Shuffling data")
sampler = np.random.permutation(len(data))
data = data.take(sampler)

# dummy encode labels, created to be used to check later the belonging to a certain class
print("Creating a dataset of indicative labels relative to the belonging CSV")
labels_full = pd.get_dummies(data['type'], prefix='type')
labels_column = data['type']
# print(labels_column)

# drop labels from training dataset
print("Deleting labels from the training dataset")
data = data.drop(columns='type')

# Feature Scaling - Z-Score Normalization
print("Scaling training set features")
data = StandardScaler().fit_transform(data)
# data = MinMaxScaler().fit_transform(data)
data = pd.DataFrame(data)

# Outliers removal - Z Score approach
# print("Removing outliers")
# filtered = (np.abs(data) < 3).all(axis=1)
# data['type'] = lab
# data = data[filtered]
# labels_full = pd.get_dummies(data['type'], prefix='type')
# data = data.drop(columns='type')
# print(data.shape)

# Outliers removal - Interquartile Range approach
# print("Removing outliers")
# Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
# IQR = Q3 - Q1
# IQR_outliers = data[((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
# data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
# data['type'] = labels_column
# labels_full = pd.get_dummies(data['type'], prefix='type')
# data = data.drop(columns='type')
# print(data.shape)

# Feature Selection
# print("Feature Selection")
# data = VarianceThreshold().fit_transform(data)
# data = SelectKBest(chi2, k=100).fit_transform(data, labels_full)
# data = pd.DataFrame(data)

# Feature Extraction
# print("Feature Extraction")
# data = PCA(50).fit_transform(data)
# data = pd.DataFrame(data)

# parsing data from Dataframe to NumPy array
# training data for the neural net
training_data = data.values

# labels for training
labels = labels_full.values


# VALIDATION

# Train/Test split - 80/20
print("Validation - 80/20 split")
x_training, x_testing, y_training, y_testing = train_test_split(training_data, labels, test_size=0.20, random_state=42)
y_training = np.argmax(y_training, axis=1)


# model = naive_bayes.BernoulliNB()
model = naive_bayes.GaussianNB()

start = time.time()
model.fit(x_training, y_training)
print("\nTraining time: " + str(time.time() - start)[0:7] + "s")

# PREDICTION
# predict genera le predizioni in output per i campioni in input
prediction = model.predict(x_testing)
print("Total time: " + str(time.time() - start)[0:7] + "s\n")


# METRICS
Metrics.metrics(y_testing, prediction, flag=1, name='Naive Bayes')

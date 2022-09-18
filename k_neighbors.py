import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import neighbors
from utils.data_setup import DataSetup
from utils.metrics import Metrics
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold

# Data retrieving
data = DataSetup.data_setup(4)

# Data preparation
# Removing all duplicates
data = data.drop_duplicates()

# Shuffling rows of dataframe, done due to consecutive dataset entry being similar to each other
print("Shuffling data")
sampler = np.random.permutation(len(data))
# i dati vengono indicizzati in base agli indici di sampler
data = data.take(sampler)

# dummy encode total_labels, store separately
# Parallel dataset is created to be used to check later the belonging to a certain class
# si crea un ulteriore dataset dove si associa 0 o 1 in relazione all'appartenza ad un csv, in base al prefisso 'type'
# ad ogni label viene aggiunto all'inizio 'type'
print("Creating a dataset of indicative total_labels relative to the belonging CSV")
labels_full = pd.get_dummies(data['type'], prefix='type')
# print(labels_full.head())

# drop total_labels from training dataset
# si eliminano le label dal dataset di training
print("Deleting 'type' column from the training dataset")
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
print("Feature Selection")
data = VarianceThreshold().fit_transform(data)
# data = SelectKBest(chi2, k=80).fit_transform(data, labels_full)
data = pd.DataFrame(data)

# Feature Extraction
# print("Feature Extraction")
# data = PCA(80).fit_transform(data)
# data = pd.DataFrame(data)

# .values trasforma i dati in formato tabellare DataFrame in un array multidimensionale NumPy
# training data for the neural net
training_data = data.values

# total_labels for training
labels = labels_full.values


# VALIDATION

# Train/Test split - 80/20
print("Validation - 80/20 split")
x_training, x_testing, y_training, y_testing = train_test_split(training_data, labels, test_size=0.20, random_state=42)

model = neighbors.KNeighborsClassifier(n_neighbors=5)

start = time.time()
model.fit(x_training, y_training)
print("\nTraining time: " + str(time.time() - start)[0:7] + "s")

# PREDICTION
# predict genera le predizioni in output per i campioni in input
prediction = model.predict(x_testing)
print("Total time: " + str(time.time() - start)[0:7] + "s\n")

# METRICS
if labels.shape[1] == 6:
    partial = 1
else:
    partial = 0
Metrics.metrics(y_testing, prediction, name='K Neighbors', partial=partial)

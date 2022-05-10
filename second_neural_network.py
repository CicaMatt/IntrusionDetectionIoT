import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from utils.data_setup import DataSetup
from utils.metrics import Metrics


# Data retrieving
data = DataSetup.data_setup(5)

# Removing all duplicates
data = data.drop_duplicates()

# Shuffling rows of dataframe
# this is done due to consecutive dataset entry being similar to each other
print("Shuffling data")
sampler = np.random.permutation(len(data))
data = data.take(sampler)

# dummy encode labels, store separately
# Parallel dataset is created to be used to check later the belonging to a certain class
# si crea un ulteriore dataset dove si associa 0 o 1 in relazione all'appartenza ad un csv, in base al prefisso 'type'
# ad ogni label viene aggiunto all'inizio 'type'
print("Creating a dataset of indicative labels relative to the belonging CSV")
labels_full = pd.get_dummies(data['type'], prefix='type')

# drop labels from training dataset
# si eliminano le label dal dataset di training
print("Deleting labels from the training dataset")
data = data.drop(columns='type')

# Feature Scaling - Z-Score Normalization
print("Scaling training set features")
data = StandardScaler().fit_transform(data)
# data = MinMaxScaler().fit_transform(data)
data = pd.DataFrame(data)


# .values restituisce una rappresentazione formato Numpy di un DataFrame, in altre parole trasforma i dati in formato tabellare in un array multidimensionale
# training data for the neural net
training_data = data.values

# labels for training
labels = labels_full.values


# test/train split  25% test
x_training, x_testing, y_training, y_testing = train_test_split(training_data, labels, test_size=0.20, random_state=42)


#second model
model = Sequential()
model.add(Dense(32, input_dim=training_data.shape[1], activation='relu'))
model.add(Dense(72, input_dim=training_data.shape[1], activation='relu'))
model.add(Dense(32, input_dim=training_data.shape[1], activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.add(Dense(labels.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,
                        patience=5, verbose=1, mode='auto')

start = time.time()
model.fit(x_training, y_training, validation_data=(x_testing, y_testing),
          callbacks=[monitor], verbose=2, epochs=100)

print("\nTraining time: " + str(time.time() - start)[0:7] + "s")

# PREDICTION
# predict genera le predizioni in output per i campioni in input
prediction = model.predict(x_testing)
print("Total time: " + str(time.time() - start)[0:7] + "s\n")

# METRICS
Metrics.metrics(y_testing, prediction)









import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
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

# dummy encode total_labels, created to be used to check later the belonging to a certain class
print("Creating a dataset of indicative total_labels relative to the belonging CSV")
labels_full = pd.get_dummies(data['type'], prefix='type')
# labels_column = data['type']
# print(labels_column)

# drop total_labels from training dataset
print("Deleting total_labels from the training dataset")
data = data.drop(columns='type')

# Feature Scaling - Z-Score Normalization
print("Scaling training set features")
data = StandardScaler().fit_transform(data)
# data = MinMaxScaler().fit_transform(data)
data = pd.DataFrame(data)

# Outliers removal - Z Score approach
# print("Removing outliers")
# filtered = (np.abs(data) < 3).all(axis=1)
# data['type'] = labels_column
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
print("Feature Extraction")
data = PCA(50).fit_transform(data)
data = pd.DataFrame(data)

# parsing data from Dataframe to NumPy array
# training data for the neural net
training_data = data.values

# total_labels for training
labels = labels_full.values

###KERAS MODEL

# VALIDATION

# Train/Test split - 80/20
print("Validation - 80/20 split")
x_training, x_testing, y_training, y_testing = train_test_split(training_data, labels, test_size=0.20, random_state=42, shuffle=True)

# create and fit model
model = Sequential()
# Dense é un tipo di layer, dove il primo valore rappresenta il numero di neuroni
# input_dim che costituisce il numero di colonne di training_data
# activation rappresenta la funzione di attivazione utilizzata
# RELU é una funzione lineare che da in output l'input diretto se esso é positivo, zero altrimenti
model.add(Dense(10, input_dim=training_data.shape[1], activation='relu'))
model.add(Dense(40, input_dim=training_data.shape[1], activation='relu'))
model.add(Dense(10, input_dim=training_data.shape[1], activation='relu'))
# kernel_initializer rappresenta l'inizializzatore della matrice dei pesi del kernel
model.add(Dense(1, kernel_initializer='normal'))
# softmax é una funzione di attivazione che converte un vettore di numeri in un vettore di probabilitá, dove
# i valori di probabilitá sono proporzionali alla relativa scala di ogni valore nel vettore
model.add(Dense(labels.shape[1], activation='softmax'))

# compile serve a configurare il modello per l'addestramento
# loss rappresenta la funzione di perdita, optimizer rappresenta l'istanza dell'optimizer
# categorical_crossentropy é usata come funzione di perdita per la classificazione multiclasse
# quando ci sono due o piú label in output
# adam é un metodo di stocastica gradiente di discesa basato sulla stima adattiva dei momenti di primo e secondo ordine
model.compile(loss='categorical_crossentropy', optimizer='adam', steps_per_execution=3)

# si istanzia un monitor che é in grado di fermare l'addestramento quando una certa metrica ha smesso di migliorare
# monitor rappresenta la quantitá da monitorare
# min_delta rappresenta il minimo cambiamento nella quantitá monitorata che va qualificata come miglioramento
# patience rappresenta il numero di epoche senza miglioramento significativo dopo le quali l'addestramento sará fermato
# verbose=1 serve a stampare messaggi di debug
# mode rappresenta la modalitá di monitoraggio della quantitá in esame
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

# fit addestra il modello per un certo numero di epoche (ovvero iterazioni sul dataset)
# primo e secondo parametro sono rispettivamente dati di input e dati obiettivo
# validation_data rappresenta i dati sui quali valutare la perdita e le metriche di ogni modello alla fine di ogni epoca
# callback contiene la lista di instanze di callback
# verbose = 2 mostra una riga di info per ogni epoca
# epochs rappresenta il numero di epoche per cui va addestrato il modello
# TRAINING
start = time.time()
model.fit(x_training, y_training, validation_data=(x_testing, y_testing),
          callbacks=[monitor], verbose=2, epochs=100)
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
Metrics.metrics(y_testing, prediction, name='Keras Neural Network', partial=partial)

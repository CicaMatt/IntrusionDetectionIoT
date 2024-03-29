import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from utils.data_setup import DataSetup

# Data retrieving
data = DataSetup.data_setup(3)
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
labels_column = data['type']
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
# total_labels for training
labels = labels_full.values

###KERAS MODEL

# VALIDATION

# K-Fold Validation - k=10
print("Validation - K-Fold Validation split (k=10)")
kf = KFold(n_splits=10, shuffle=True)
print(kf.get_n_splits())
i = 1

accuracy_score_max = 0
precision_score_max = 0
recall_score_max = 0
f1_score_max = 0

for training_index, testing_index in kf.split(training_data):
    # print("TRAIN:", training_index, "TEST:", testing_index)
    x_training, x_testing = training_data[training_index], training_data[testing_index]
    y_training, y_testing = labels[training_index], labels[testing_index]

    print("\nFit n." + i.__str__())
    # create and fit model
    model = Sequential()
    # il metodo add aggiunge layer alla rete neurale
    # Dense é un tipo di layer, dove il primo valore rappresenta il numero di neuroni
    # input_dim che costituisce il numero di colonne di training_data, mentre activation rappresenta la funzione di attivazione utilizzata
    # RELU é una funzione lineare che da in output l'input diretto se esso é positivo, zero altrimenti
    model.add(Dense(10, input_dim=training_data.shape[1], activation='relu'))
    model.add(Dense(40, input_dim=training_data.shape[1], activation='relu'))
    model.add(Dense(10, input_dim=training_data.shape[1], activation='relu'))
    # kernel_initializer rappresenta l'inizializzatore della matrice dei pesi del kernel
    model.add(Dense(1, kernel_initializer='normal'))
    # softmax é una funzione di attivazione che converte un vettore di numeri in un vettore di probabilitá, dove
    # i valori di probabilitá sono proporzionali alla reliva scala di ogni valore nel vettore
    model.add(Dense(labels.shape[1], activation='softmax'))

    # compile serve a configurare il modello per l'addestramento
    # loss rappresenta la funzione di perdita, optimizer rappresenta l'istanza dell'optimizer
    # categorical_crossentropy é usata come funzione di perdita per la classificazione multiclasse
    # quando ci sono due o piú label in output
    # adam é un metodo di stocastica gradiente di discesa basato sulla stima adattiva dei momenti di primo ordine e secondo ordine
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # si istanzia un monitor che é in grado di fermare l'addestramento quando una certa metrica ha smesso di migliorare
    # monitor rappresenta la quantitá da monitorare
    # min_delta rappresenta il minimo cambiamento nella quantitá monitorata che va qualificata come miglioramento
    # patience rappresenta il numero di epoche senza miglioramento dopo le quali l'addestramento sará fermato
    # verbose=1 serve a stampare messaggi di debug
    # mode rappresenta la modalitá di monitoraggio della quantitá in esame
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

    # fit addestra il modello per un certo numero di periodi (ovvero iterazioni sul dataset)
    # primo e secondo parametro sono rispettivamente dati di input e dati obiettivo
    # validation_data rappresenta i dati sui quali valutare la perdina e le metriche di ogni modello alla fine di ogni epoca
    # callback contiene la lista di instanze di callback
    # verbose = 2 mostra una riga di info per ogni epoca
    # epochs rappresenta il numero di epoche per cui va addestrato il modello
    start = time.time()
    model.fit(x_training, y_training, validation_data=(x_testing, y_testing),
              callbacks=[monitor], verbose=2, epochs=50)
    end = time.time()
    print("Training time (Fit " + i.__str__() + "): " + str(end - start)[0:6] + "s " + i.__str__())

    # PREDICTION
    # predict genera le predizioni in output per i campioni in input
    prediction = model.predict(x_testing)
    print("Total time (Fit " + i.__str__() + "): " + str(time.time() - start)[0:7] + "s\n")

    # METRICS
    # argmax restituisce gli indici dei valori massimi lungo un asse
    prediction = np.argmax(prediction, axis=1)
    truth = np.argmax(y_testing, axis=1)
    accuracy_score = metrics.accuracy_score(truth, prediction)
    precision_score = metrics.precision_score(truth, prediction, average='weighted', zero_division=0)
    recall_score = metrics.recall_score(truth, prediction, average='weighted')
    f1_score = metrics.f1_score(truth, prediction, average="weighted")
    confusion_matrix = metrics.confusion_matrix(truth, prediction)

    print("Accuracy (Fit " + i.__str__() + "): " + "{:.2%}".format(float(accuracy_score)))
    print("Precision (Fit " + i.__str__() + "): " + "{:.2%}".format(float(precision_score)))
    print("Recall (Fit " + i.__str__() + "): " + "{:.2%}".format(float(recall_score)))
    print("F1 (Fit " + i.__str__() + "): " + "{:.2%}".format(float(f1_score)))
    print("Confusion Matrix (Fit " + i.__str__() + "):")
    print(confusion_matrix)

    if accuracy_score_max < accuracy_score:
        accuracy_score_max = accuracy_score
        precision_score_max = precision_score
        recall_score_max = recall_score
        f1_score_max = f1_score
    i += 1

print("\n\nAccuracy (Max): " + "{:.2%}".format(float(accuracy_score_max)))
print("Precision (Max): " + "{:.2%}".format(float(precision_score_max)))
print("Recall (Max): " + "{:.2%}".format(float(recall_score_max)))
print("F1 (Max): " + "{:.2%}".format(float(f1_score_max)))

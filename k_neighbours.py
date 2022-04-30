import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics, neighbors


# reading CSV files based on given 'set' and instantiate a DataFrame
set = "5"
print("CSV file reading")
benign = pd.read_csv('archive/' + set + '.benign.csv')
g_c = pd.read_csv('archive/' + set + '.gafgyt.combo.csv')
g_j = pd.read_csv('archive/' + set + '.gafgyt.junk.csv')
g_s = pd.read_csv('archive/' + set + '.gafgyt.scan.csv')
g_t = pd.read_csv('archive/' + set + '.gafgyt.tcp.csv')
g_u = pd.read_csv('archive/' + set + '.gafgyt.udp.csv')
m_a = pd.read_csv('archive/' + set + '.mirai.ack.csv')
m_sc = pd.read_csv('archive/' + set + '.mirai.scan.csv')
m_sy = pd.read_csv('archive/' + set + '.mirai.syn.csv')
m_u = pd.read_csv('archive/' + set + '.mirai.udp.csv')
m_u_p = pd.read_csv('archive/' + set + '.mirai.udpplain.csv')

# randomly sampling the DataFrames by n=amount entries
amount = 15000
print("DataFrame sampling")
benign = benign.sample(n=amount, replace=False)
g_c = g_c.sample(n=amount, replace=False)
g_j = g_j.sample(n=amount, replace=False)
g_s = g_s.sample(n=amount, replace=False)
g_t = g_t.sample(n=amount, replace=False)
g_u = g_u.sample(n=amount, replace=False)
m_a = m_a.sample(n=amount, replace=False)
m_sc = m_sc.sample(n=amount, replace=False)
m_sy = m_sy.sample(n=amount, replace=False)
m_u = m_u.sample(n=amount, replace=False)
m_u_p = m_u_p.sample(n=amount, replace=False)

# adding labels to each DataFrame
benign['type'] = 'benign'
m_u['type'] = 'mirai_udp'
g_c['type'] = 'gafgyt_combo'
g_j['type'] = 'gafgyt_junk'
g_s['type'] = 'gafgyt_scan'
g_t['type'] = 'gafgyt_tcp'
g_u['type'] = 'gafgyt_udp'
m_a['type'] = 'mirai_ack'
m_sc['type'] = 'mirai_scan'
m_sy['type'] = 'mirai_syn'
m_u_p['type'] = 'mirai_udpplain'

# concatena tutti i dizionari ottenuti
# sort=False non mischia i dati fra loro
# ignore_index=True non usa i valori degli indici per la concatenazione, quindi l'asse risultante andrá da 0 a n-1
print("Concatenating all DataFrames into one")
data = pd.concat([benign, m_u, g_c, g_j, g_s, g_t, g_u, m_a, m_sc, m_sy, m_u_p], sort=False, ignore_index=True)

# Shuffling rows of dataframe, done due to consecutive dataset entry being similar to each other
print("Shuffling data")
sampler = np.random.permutation(len(data))
# i dati vengono indicizzati in base agli indici di sampler
data = data.take(sampler)

# dummy encode labels, store separately
# Parallel dataset is created to be used to check later the belonging to a certain class
# si crea un ulteriore dataset dove si associa 0 o 1 in relazione all'appartenza ad un csv, in base al prefisso 'type'
# ad ogni label viene aggiunto all'inizio 'type'
print("Creating a dataset of indicative labels relative to the belonging CSV")
labels_full = pd.get_dummies(data['type'], prefix='type')
# print(labels_full.head())

# drop labels from training dataset
# si eliminano le label dal dataset di training
print("Deleting labels from the training dataset")
data = data.drop(columns='type')

# Feature Scaling - Z-Score Normalization
print("Scaling training set features")
for i in data.columns:
    data[i] = zscore(data[i])
# MinMaxScaler().fit_transform(data_st)
# print(data_st.head(50))
# print(data_st.tail(50))


# .values trasforma i dati in formato tabellare DataFrame in un array multidimensionale NumPy
# training data for the neural net
training_data = data.values

# labels for training
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
# argmax restituisce gli indici dei valori massimi lungo un asse
prediction = np.argmax(prediction, axis=1)
truth = np.argmax(y_testing, axis=1)
accuracy_score = metrics.accuracy_score(truth, prediction)
precision_score = metrics.precision_score(truth, prediction, average='weighted', zero_division=0)
recall_score = metrics.recall_score(truth, prediction, average='weighted')
f1_score = metrics.f1_score(truth, prediction, average="weighted")
print("Accuracy: "+"{:.2%}".format(float(accuracy_score)))
print("Precision: "+"{:.2%}".format(float(precision_score)))
print("Recall: "+"{:.2%}".format(float(recall_score)))
print("F1: "+"{:.2%}".format(float(f1_score)))

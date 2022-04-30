import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# reading CSV files based on given 'set' and instantiate a DataFrame
set = "1"
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

# Shuffling rows of dataframe
# this is done due to consecutive dataset entry being similar to each other
print("Shuffling data")
sampler = np.random.permutation(len(data))
# i dati vengono indicizzati in base all'effettiva posizione nella lista e non in base al valore dell'indice
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

data_st = data.copy()

# Feature Scaling - Z-Score Normalization
print("Scaling training set features")
for i in data.columns:
    data_st[i] = zscore(data[i])
# MinMaxScaler().fit_transform(data_st)
# print(data_st.head(50))
# print(data_st.tail(50))


# .values restituisce una rappresentazione formato Numpy di un DataFrame, in altre parole trasforma i dati in formato tabellare in un array multidimensionale
# training data for the neural net
train_data_st = data_st.values

# labels for training
labels = labels_full.values


# test/train split  25% test
x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(
    train_data_st, labels, test_size=0.25, random_state=42)

#second model
model2 = Sequential()
model2.add(Dense(32, input_dim=train_data_st.shape[1], activation='relu'))
model2.add(Dense(72, input_dim=train_data_st.shape[1], activation='relu'))
model2.add(Dense(32, input_dim=train_data_st.shape[1], activation='relu'))
model2.add(Dense(1, kernel_initializer='normal'))
model2.add(Dense(labels.shape[1],activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,
                        patience=5, verbose=1, mode='auto')
model2.fit(x_train_st,y_train_st,validation_data=(x_test_st,y_test_st),
          callbacks=[monitor],verbose=2,epochs=100)

# metrics
pred_st = model2.predict(x_test_st)
pred_st = np.argmax(pred_st,axis=1)
y_eval_st = np.argmax(y_test_st,axis=1)
score_st = metrics.accuracy_score(y_eval_st, pred_st)
print("accuracy: {}".format(score_st))








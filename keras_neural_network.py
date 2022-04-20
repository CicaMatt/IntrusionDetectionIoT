import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.callbacks import EarlyStopping

#legge i csv e restituisce un DataFrame o TextParser
benign=pd.read_csv('archive/5.benign.csv')
g_c=pd.read_csv('archive/5.gafgyt.combo.csv')
g_j=pd.read_csv('archive/5.gafgyt.junk.csv')
g_s=pd.read_csv('archive/5.gafgyt.scan.csv')
g_t=pd.read_csv('archive/5.gafgyt.tcp.csv')
g_u=pd.read_csv('archive/5.gafgyt.udp.csv')
m_a=pd.read_csv('archive/5.mirai.ack.csv')
m_sc=pd.read_csv('archive/5.mirai.scan.csv')
m_sy=pd.read_csv('archive/5.mirai.syn.csv')
m_u=pd.read_csv('archive/5.mirai.udp.csv')
m_u_p=pd.read_csv('archive/5.mirai.udpplain.csv')
print("file csv letti")

#campiona in modo casuale i csv letti
#frac rappresenta la percentuale di campioni presi sul totale, replace=False non permette di selezionare la stessa riga
benign=benign.sample(frac=0.25,replace=False)
g_c=g_c.sample(frac=0.25,replace=False)
g_j=g_j.sample(frac=0.5,replace=False)
g_s=g_s.sample(frac=0.5,replace=False)
g_t=g_t.sample(frac=0.15,replace=False)
g_u=g_u.sample(frac=0.15,replace=False)
m_a=m_a.sample(frac=0.25,replace=False)
m_sc=m_sc.sample(frac=0.15,replace=False)
m_sy=m_sy.sample(frac=0.25,replace=False)
m_u=m_u.sample(frac=0.1,replace=False)
m_u_p=m_u_p.sample(frac=0.27,replace=False)

#si aggiunge la coppia   tipo di dato/file csv originale   ai dizionari ottenuti
benign['type']='benign'
m_u['type']='mirai_udp'
g_c['type']='gafgyt_combo'
g_j['type']='gafgyt_junk'
g_s['type']='gafgyt_scan'
g_t['type']='gafgyt_tcp'
g_u['type']='gafgyt_udp'
m_a['type']='mirai_ack'
m_sc['type']='mirai_scan'
m_sy['type']='mirai_syn'
m_u_p['type']='mirai_udpplain'

#concatena tutti i dizionari ottenuti
#sort=False non mischia i dati fra loro
#ignore_index=True non usa i valori degli indici per la concatenazione, quindi l'asse risultante andrá da 0 a n-1
data=pd.concat([benign, m_u, g_c, g_j, g_s, g_t, g_u, m_a, m_sc, m_sy, m_u_p],
             sort=False, ignore_index=True)

#how many instances of each class
#raggruppa i dati in base all'etichetta type
data.groupby('type')['type'].count()

#shuffle rows of dataframe
#permuta in modo casuale i dati
sampler=np.random.permutation(len(data))
#i dati vengono indicizzati in base all'effettiva posizione nella lista
data=data.take(sampler)
#data viene sovrascritto con i suoi soli 5 primi campioni
data.head()

#dummy encode labels, store separately
#a partire dai dati, si convertono le variabili categoriche in variabili indicative, in base al prefisso 'type'
labels_full=pd.get_dummies(data['type'], prefix='type')
print(labels_full)
#labels_full viene sovrascritto con i suoi soli 5 primi campioni
labels_full.head()

#drop labels from training dataset
#si eliminano le label dal dataset di training
data=data.drop(columns='type')
data.head()

#standardize numerical columns
def standardize(df,col):
    df[col]= (df[col]-df[col].mean())/df[col].std()

data_st=data.copy()
#la funzione iloc permette di selezionare una particolare cella del dataset, selezionando un valore che appartiene
# ad una certa riga o colonna dai dati di partenza
for i in (data_st.iloc[:,:-1].columns):
    standardize (data_st,i)

data_st.head()

# .values restituisce una rappresentazione formato Numpy di un DataFrame, in altre parole trasforma i dati in formato tabellare in un array multidimensionale
#training data for the neural net
train_data_st=data_st.values

#labels for training
labels=labels_full.values

#Keras model
# test/train split  25% test
x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(
    train_data_st, labels, test_size=0.25, random_state=42)

#  create and fit model
model = Sequential()
#il metodo add aggiunge layer alla rete neurale
#Dense é un tipo di layer, dove il primo valore rappresenta la dimensionalitá dello spazio di output (numero di neuroni dell'ultimo livello?)
#input_dim che costituisce il numero di righe e colonne di train_data_st, mentre activation rappresenta la funzione di attivazione utilizzata
#RELU é una funzione lineare che da in ouput l'input diretto se esso é positivo, zero altrimenti
model.add(Dense(10, input_dim=train_data_st.shape[1], activation='relu'))
model.add(Dense(40, input_dim=train_data_st.shape[1], activation='relu'))
model.add(Dense(10, input_dim=train_data_st.shape[1], activation='relu'))
#kernel_initializer rappresenta l'inizializzatore della matrice dei pesi del kernel
model.add(Dense(1, kernel_initializer='normal'))
#softmax é una funzione di attivazione che converte un vettore di numeri in un vettore di probabilitá, dove
#i valori di probabilitá sono proporzionali alla reliva scala di ogni valore nel vettore
model.add(Dense(labels.shape[1],activation='softmax'))

#compile serve a configurare il modello per l'addestramento
#loss rappresenta la funzione di perdita, optimizer rappresenta l'istanza dell'optimizer
#categorical_crossentropy é usata come funzione di perdita per la classificazione multiclasse
#quando ci sono due o piú label in output
#adam é un metodo di stocastica gradiente di discesa basato sulla stima adattiva dei momenti di primo ordine e secondo ordine
model.compile(loss='categorical_crossentropy', optimizer='adam')

#si istanzia un monitor che é in grado di fermare l'addestramento quando una certa metrica ha smesso di migliorare
#monitor rappresenta la quantitá da monitorare
#min_delta rappresenta il minimo cambiamento nella quantitá monitorata che va qualificata come miglioramento
#patience rappresenta il numero di epoche senza miglioramento dopo le quali l'addestramento sará fermato
#verbose=1 serve a stampare messaggi di debug
#mode rappresenta la modalitá di monitoraggio della quantitá in esame
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,
                        patience=5, verbose=1, mode='auto')

#fit addestra il modello per un certo numero di periodi (ovvero iterazioni sul dataset)
#primo e secondo parametro sono rispettivamente dati di input e dati obiettivo
#validation_data rappresenta i dati sui quali valutare la perdina e le metriche di ogni modello alla fine di ogni epoca
#callback contiene la lista di instanze di callback
#verbose = 2 mostra una riga di info per ogni epoca
#epochs rappresenta il numero di epoche per cui va addestrato il modello
model.fit(x_train_st,y_train_st,validation_data=(x_test_st,y_test_st),
          callbacks=[monitor],verbose=2,epochs=500)

# metrics
pred_st = model.predict(x_test_st)
pred_st = np.argmax(pred_st,axis=1)
y_eval_st = np.argmax(y_test_st,axis=1)
score_st = metrics.accuracy_score(y_eval_st, pred_st)
print("accuracy: {}".format(score_st))









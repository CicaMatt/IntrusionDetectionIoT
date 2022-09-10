import pandas as pd
import seaborn
from matplotlib import pyplot as plt

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
amount = 1000
minor_amount = int(20000)
print("DataFrame sampling")
benign = benign.sample(n=amount, replace=False)
g_c = g_c.sample(n=amount, replace=False)
g_j = g_j.sample(n=amount, replace=False)
g_s = g_s.sample(n=amount, replace=False)
g_t = g_t.sample(n=amount, replace=False)
g_u = g_u.sample(n=amount, replace=False)
if set != 3 and set != 7:
    m_a = m_a.sample(n=amount, replace=False)
    m_sc = m_sc.sample(n=amount, replace=False)
    m_sy = m_sy.sample(n=amount, replace=False)
    m_u = m_u.sample(n=amount, replace=False)
    m_u_p = m_u_p.sample(n=amount, replace=False)

# adding total_labels to each DataFrame
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


# Correlation matrix
# print("Correlation matrix:")
# correlation_matrix = data.corr()
# plt.subplots(figsize=(35, 25))
# seaborn.heatmap(correlation_matrix, vmax=.8, square=True)
# plt.show()

# Scatter plot
print("Scatter plot:")
seaborn.set()
columns = ['MI_dir_L0.1_weight', 'H_L0.1_weight', 'HH_L0.1_weight', 'HH_jit_L0.1_weight', 'HpHp_L0.1_weight']
seaborn.pairplot(data[columns], height=2.5)
# seaborn.pairplot(data[list(data.columns)].sample(n=10, replace=False), height=2.5)
plt.show()

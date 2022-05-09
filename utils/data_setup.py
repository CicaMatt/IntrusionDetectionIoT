import pandas as pd

class DataSetup:
    def data_setup(set):
        # reading CSV files based on given 'set' and instantiate a DataFrame
        print("CSV file reading")
        benign = pd.read_csv('archive/' + str(set) + '.benign.csv')
        gafgyt_combo = pd.read_csv('archive/' + str(set) + '.gafgyt.combo.csv')
        gafgyt_junk = pd.read_csv('archive/' + str(set) + '.gafgyt.junk.csv')
        gafgyt_scan = pd.read_csv('archive/' + str(set) + '.gafgyt.scan.csv')
        gafgyt_tcp = pd.read_csv('archive/' + str(set) + '.gafgyt.tcp.csv')
        gafgyt_udp = pd.read_csv('archive/' + str(set) + '.gafgyt.udp.csv')
        mirai_ack = pd.read_csv('archive/' + str(set) + '.mirai.ack.csv')
        mirai_scan = pd.read_csv('archive/' + str(set) + '.mirai.scan.csv')
        mirai_syn = pd.read_csv('archive/' + str(set) + '.mirai.syn.csv')
        mirai_udp = pd.read_csv('archive/' + str(set) + '.mirai.udp.csv')
        mirai_udpplain = pd.read_csv('archive/' + str(set) + '.mirai.udpplain.csv')

        # randomly sampling the DataFrames by n=amount entries
        amount = 15000
        print("DataFrame sampling")
        benign = benign.sample(n=amount, replace=False)
        gafgyt_combo = gafgyt_combo.sample(n=amount, replace=False)
        gafgyt_junk = gafgyt_junk.sample(n=amount, replace=False)
        gafgyt_scan = gafgyt_scan.sample(n=amount, replace=False)
        gafgyt_tcp = gafgyt_tcp.sample(n=amount, replace=False)
        gafgyt_udp = gafgyt_udp.sample(n=amount, replace=False)
        mirai_ack = mirai_ack.sample(n=amount, replace=False)
        mirai_scan = mirai_scan.sample(n=amount, replace=False)
        mirai_syn = mirai_syn.sample(n=amount, replace=False)
        mirai_udp = mirai_udp.sample(n=amount, replace=False)
        mirai_udpplain = mirai_udpplain.sample(n=amount, replace=False)

        # adding labels to each DataFrame
        benign['type'] = 'benign'
        mirai_udp['type'] = 'mirai_udp'
        gafgyt_combo['type'] = 'gafgyt_combo'
        gafgyt_junk['type'] = 'gafgyt_junk'
        gafgyt_scan['type'] = 'gafgyt_scan'
        gafgyt_tcp['type'] = 'gafgyt_tcp'
        gafgyt_udp['type'] = 'gafgyt_udp'
        mirai_ack['type'] = 'mirai_ack'
        mirai_scan['type'] = 'mirai_scan'
        mirai_syn['type'] = 'mirai_syn'
        mirai_udpplain['type'] = 'mirai_udpplain'

        # concatena tutti i dizionari ottenuti
        # sort=False non mischia i dati fra loro
        # ignore_index=True non usa i valori degli indici per la concatenazione, quindi l'asse risultante andrá da 0 a n-1
        print("Concatenating all DataFrames into one")
        data = pd.concat(
            [benign, mirai_udp, gafgyt_combo, gafgyt_junk, gafgyt_scan, gafgyt_tcp,
             gafgyt_udp, mirai_ack, mirai_scan, mirai_syn, mirai_udpplain],
            sort=False, ignore_index=True)

        return data
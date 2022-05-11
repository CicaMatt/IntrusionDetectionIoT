import pandas as pd
import os

set=1
gafgyt_tcp = pd.read_csv('archive/' + str(set) + '.gafgyt.tcp.csv')
gafgyt_udp = pd.read_csv('archive/' + str(set) + '.gafgyt.udp.csv')

gafgyt_tcp = gafgyt_tcp.drop_duplicates()
gafgyt_udp = gafgyt_udp.drop_duplicates()
first = pd.DataFrame(gafgyt_tcp)
second = pd.DataFrame(gafgyt_udp)
first.to_csv('new/one.csv', index=False, header=False)
second.to_csv('new/two.csv', index=False, header=False)



# %% Show all database
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data_df = pd.read_csv("Darknet.csv")
data_df


# %% Show columns
data_df.columns

# %% TODO

# 2. Parametry - Kluczowe (2-4)
#       Okres czasu
#       Parametr od czasu
#       Czy pomiar jest ciągły?
#       Pojedyncze czy regularne przerwy?
# 3. Statystyki takie jak:
#       Max
#       Min
#       Średnia
# 4. Histogramy

# %% Rozbicie Timestamp

data_df.Timestamp
data_df["Timestamp"] = pd.to_datetime(data_df.Timestamp)
data_df["Timestamp"]

data_df['year'] = pd.DatetimeIndex(data_df.Timestamp).year
data_df['month'] = pd.DatetimeIndex(data_df.Timestamp).month
data_df['day'] = pd.DatetimeIndex(data_df.Timestamp).day
data_df['hour'] = pd.DatetimeIndex(data_df.Timestamp).hour
data_df['minute'] = pd.DatetimeIndex(data_df.Timestamp).minute
data_df['weekday'] = pd.DatetimeIndex(data_df.Timestamp).weekday


# %% PODZIAL ZE WZGLEDU NA MIESIACE
data_df_styczen = data_df[data_df.month == 1]
data_df_luty = data_df[data_df.month == 2]
data_df_marzec = data_df[data_df.month == 3]
data_df_kwiecien = data_df[data_df.month == 4]
data_df_maj = data_df[data_df.month == 5]
data_df_lipiec = data_df[data_df.month == 7]
data_df_sierpien = data_df[data_df.month == 8]
data_df_wrzesien = data_df[data_df.month == 9]

data_df_2015 = data_df[data_df.year == 2015]
data_df_2016 = data_df[data_df.year == 2016]

# %% Miesiace w ktorych byly prowadzone badania
data_df_2015['month'].unique()              # 1,2,3,4,5,7,8,9
data_df_2016['month'].unique()              # 2


# %% Godziny w ktorych byly prowadzone badania
data_df_2015['hour'].unique()               # 9,10,11,12,13,14,15,16,17,18
data_df_2016['hour'].unique()               # 8,9,10,11,12,13,14,15,17,18


# %% Dni tygodnia w ktorych byly prowadzone badania
data_df_2015['weekday'].unique()            # 0,1,2,3,4,5,6
data_df_2016['weekday'].unique()            # 1,2,3


# %%
Timestamp_df = data_df_2015[["month", "CWE Flag Count", ]]
ax2 = Timestamp_df.plot.scatter(x='month', y='CWE Flag Count')

# %% PODZIAL ZE WZGLEDU NA TYP POŁĄCZENIA

chat_df = data_df.loc[data_df["Label.1"] == "Chat"]
AUDIO_STREAMING_df = data_df.loc[data_df["Label.1"] == "AUDIO-STREAMING"]
Browsing_df = data_df.loc[data_df["Label.1"] == "Browsing"]
Email_df = data_df.loc[data_df["Label.1"] == "Email"]
File_Transfer_df = data_df.loc[data_df["Label.1"] == "File-Transfer"]
File_transfer_df = data_df.loc[data_df["Label.1"] == "File-transfer"]
Video_Streaming_df = data_df.loc[data_df["Label.1"] == "Video-Streaming"]
Audio_Streaming_df = data_df.loc[data_df["Label.1"] == "Audio-Streaming"]
Video_streaming_df = data_df.loc[data_df["Label.1"] == "Video-streaming"]
VOIP_df = data_df.loc[data_df["Label.1"] == "VOIP"]


# %% PODZIAL ZE WZGLĘDU NA TECHNOLOGIE
Non_Tor_df = data_df.loc[data_df["Label"] == "Non_Tor"]
NonVPN_df = data_df.loc[data_df["Label"] == "NonVPN"]
Tor_df = data_df.loc[data_df["Label"] == "Tor"]
VPN_df = data_df.loc[data_df["Label"] == "VPN"]


# %% HISTOGRAMY

plt.xticks(rotation=0)

sns.set_theme(style="whitegrid")
typ_polaczenia_df = data_df["Label.1"]
technologie_df = data_df["Label"]


b = sns.histplot(data_df["Label"]).set(title="Technologia")
g = sns.histplot(data=typ_polaczenia_df).set(title="Typ połączenia")


# %% Unikalne IP

data_df_2015.describe()

# %% OGOLNIE

#######################################################################

# %% Zliczenie połączeń z danego IP SOURCE
data_df_2015.groupby(['Src IP']).size()

# %% Zliczenie połączeń z danego IP DESTINATION
data_df_2015.groupby(['Dst IP']).size()

# %%  TYP POLACZENIA SRC IP

#######################################################################
# %%

chat_df.groupby(['Src IP']).size()

# %%
AUDIO_STREAMING_df.groupby(['Src IP']).size()

# %%
Browsing_df.groupby(['Src IP']).size()

# %%
Email_df.groupby(['Src IP']).size()

# %%
File_Transfer_df.groupby(['Src IP']).size()

# %%
Video_Streaming_df.groupby(['Src IP']).size()

# %%
Audio_Streaming_df.groupby(['Src IP']).size()

# %%
Video_streaming_df.groupby(['Src IP']).size()

# %%
VOIP_df.groupby(['Src IP']).size()

# %% TYP POLACZENIA DST IP

#######################################################################
# %%
chat_df.groupby(['Dst IP']).size()

# %%
AUDIO_STREAMING_df.groupby(['Dst IP']).size()

# %%
Browsing_df.groupby(['Dst IP']).size()

# %%
Email_df.groupby(['Dst IP']).size()

# %%
File_Transfer_df.groupby(['Dst IP']).size()

# %%
Video_Streaming_df.groupby(['Dst IP']).size()

# %%
Audio_Streaming_df.groupby(['Dst IP']).size()

# %%
Video_streaming_df.groupby(['Dst IP']).size()

# %%
VOIP_df.groupby(['Dst IP']).size()

# %% TECHNOLOGIA DST IP
#######################################################################
Non_Tor_df.groupby(['Dst IP']).size()

# %%
NonVPN_df.groupby(['Dst IP']).size()

# %%
Tor_df.groupby(['Dst IP']).size()

# %%
VPN_df.groupby(['Dst IP']).size()

# %% TECHNOLOGIA SRC IP
#######################################################################
Non_Tor_df.groupby(['Src IP']).size()

# %%
NonVPN_df.groupby(['Src IP']).size()

# %%
Tor_df.groupby(['Src IP']).size()

# %%
VPN_df.groupby(['Src IP']).size()[0]

# %%
test = VPN_df.groupby(['Src IP']).size()
test
# %%
import ast
from collections import Counter

f = open("dest_ip_fix.txt", "r", encoding="utf8")

for line in f:
    line = ast.literal_eval(line)
    if "country" in line:
        data_df.loc[data_df['Dst IP'] == line['query'], 'Dst IP'] = line['country']


# %%
data_df['Dst IP']

# %%
data_df.groupby(['Dst IP']).size().sort_values(ascending=False).head(20)









# %%
import ast
from collections import Counter

f = open("source_ip_fix.txt", "r", encoding="utf8")

for line in f:
    line = ast.literal_eval(line)
    if "country" in line:
        data_df.loc[data_df['Src IP'] == line['query'], 'Src IP'] = line['country']


# %%
data_df['Src IP']


# %%
data_df.groupby(['Src IP']).size().sort_values(ascending=False).head(20)
#%%

data_df.groupby(['Src IP', 'Dst IP']).size().sort_values(ascending=False).head(20)

# %%

sum_df = data_df.groupby(['Src IP', 'Dst IP']).agg({'Flow Duration': 'mean'})
sum_df.sort_values(by="Flow Duration", ascending=False).head(20)

# %%

import requests
import time
import json
from collections import Counter
# %% LOKALIZACJA IP



file = open("source_ip.txt", encoding="utf8")
# %%
plik = file.readlines()
for line in plik: 
	print(line)

# %%

ip_location = []
for ip in file:
  time.sleep(1)
  print(ip+"\n")
  response = requests.get("http://ip-api.com/json/%22+str(ip)).json()")
  ip_location.append(response)
  print(response)
# %% HISTOGRAMY
# print(ip_location['query'])
cities = []
for i in file[1:]:
  if 'country' in i:
    print(i['country'])
    cities.append(i['country'])
	
Counter(cities)

# %% zapis do pliku
file = open("source_ip.txt", "w+")
file.write(str(ip_location))
file.close()
# %% OGOLNIE
dest_ip_location = []
for ip in dest_ip_location:
  time.sleep(0.5)
  print(ip+"\n")
  response = requests.get("http://ip-api.com/json/%22+str(ip)).json()")
  dest_ip_location.append(response)
  print(response)
#######################################################################
































# %% JEDEN PSH
Flow_Duration_df = data_df_2015[["Flow Duration", "PSH Flag Count", ]]
ax2 = Flow_Duration_df.plot.scatter(x='Flow Duration', y='PSH Flag Count')

# %% DRUGI
Flow_Duration_df = data_df_2015[["Flow Duration", "SYN Flag Count", ]]
ax2 = Flow_Duration_df.plot.scatter(x='Flow Duration', y='SYN Flag Count')

# %% TRZECI
Flow_Duration_df = data_df_2015[["Flow Duration", "ACK Flag Count", ]]
ax2 = Flow_Duration_df.plot.scatter(x='Flow Duration', y='ACK Flag Count')

# %% CZWARTY
Flow_Duration_df = data_df_2015[["Flow Duration", "URG Flag Count", ]]
ax2 = Flow_Duration_df.plot.scatter(x='Flow Duration', y='URG Flag Count')

# %% PIATY
Flow_Duration_df = data_df_2015[["Flow Duration", "CWE Flag Count", ]]
ax2 = Flow_Duration_df.plot.scatter(x='Flow Duration', y='CWE Flag Count')

# %% SZOSTY
Flow_Duration_df = data_df_2015[["Flow Duration", "ECE Flag Count", ]]
ax2 = Flow_Duration_df.plot.scatter(x='Flow Duration', y='ECE Flag Count')

# %% SIODMY
Flow_Duration_df = data_df_2015[["Flow Duration", "RST Flag Count", ]]
ax2 = Flow_Duration_df.plot.scatter(x='Flow Duration', y='RST Flag Count')

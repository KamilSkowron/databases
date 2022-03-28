#%% Show all database
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data_df = pd.read_csv("Darknet.csv")
data_df


# %% Show columns
data_df.columns

#%% TODO

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

#%% Rozbicie Timestamp

data_df.Timestamp
data_df["Timestamp"] = pd.to_datetime(data_df.Timestamp)
data_df["Timestamp"]

data_df['year'] = pd.DatetimeIndex(data_df.Timestamp).year
data_df['month'] = pd.DatetimeIndex(data_df.Timestamp).month
data_df['day'] = pd.DatetimeIndex(data_df.Timestamp).day
data_df['hour'] = pd.DatetimeIndex(data_df.Timestamp).hour
data_df['minute'] = pd.DatetimeIndex(data_df.Timestamp).minute
data_df['weekday'] = pd.DatetimeIndex(data_df.Timestamp).weekday


#%% Nowe tabelki dla łatwiejszej analizy
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

#%% Miesiace w ktorych byly prowadzone badania
data_df_2015['month'].unique()              # 1,2,3,4,5,7,8,9
data_df_2016['month'].unique()              # 2


#%% Godziny w ktorych byly prowadzone badania
data_df_2015['hour'].unique()               # 9,10,11,12,13,14,15,16,17,18
data_df_2016['hour'].unique()               # 8,9,10,11,12,13,14,15,17,18    


#%% Dni tygodnia w ktorych byly prowadzone badania
data_df_2015['weekday'].unique()            # 0,1,2,3,4,5,6
data_df_2016['weekday'].unique()            # 1,2,3


#%% Total Bwd packets
subflowBwdBytes_df = data_df_2015[["Timestamp","Total Bwd packets"]]
ax1 = subflowBwdBytes_df.plot.scatter(x = 'Timestamp', y = 'Total Bwd packets')


#%% Total Fwd Packet
subflowFwdBytes_df = data_df_2015[["Timestamp","Total Fwd Packet"]]
ax2 = subflowFwdBytes_df.plot.scatter(x = 'Timestamp', y = 'Total Fwd Packet')














# %% Comparison of "Protocol" and "Average Packet Size"

data_df[["Protocol","Average Packet Size"]]
plt.scatter(data_df["Protocol"],data_df["Average Packet Size"])
#plt.hist(data_df["Average Packet Size"],bins=np.arange(0,400,10))
#plt.bar(data_df["Protocol"],data_df["Average Packet Size"])

# %%

#data_df[["Flow ID","Average Packet Size"]]
#plt.scatter(data_df["Flow ID"],data_df["Average Packet Size"])
#plt.hist(data_df["Active Max"])
plt.bar(data_df["Timestamp"],data_df["Flow ID"])
#plt.bar(data_df["Protocol"],data_df["Average Packet Size"])
# %%

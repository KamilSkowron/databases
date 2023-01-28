import numpy as np
import pandas as pd

data = {'X1': [155, 15, 13, 11, 12, 11, 15, 156, 13, 10, 12],
        'X2': [149, 169, 172, 148, 156, 162, 189, 555, 147, 150, 153]}

lista = [155, 15, 13, 11, 12, 11, 15, 156, 13, 10, 12]
lista2 = [149, 169, 172, 148, 156, 162, 189, 555, 147,
          150, 153, 155, 15, 13, 11, 12, 11, 15, 156, 13, 10, 12]


df = pd.DataFrame(data)

res = []

u = np.average(df)
o = df.values.std()


for date, row in df.T.iteritems():
    print(date,row)
    for i, v in enumerate(row):
        
        x = v
        z = (x - u)/o

        if z > 3:
            res.append((date,i))
print (res)

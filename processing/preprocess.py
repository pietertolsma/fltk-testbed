import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("D:/logging/results/aggregated data.csv")
print(df)
df = df.drop(columns=["time(ms)", "training loss"])
df = df.rename(columns={"time_level(ms)": "time_level", "network depth": "layers", "data_parallelism": "nodes"})
df = df.replace({"time_level": {300000 * i: i for i in range(1, 5)},
                 "cores": {"1000m": 1, "2000m": 2},
                 "batch_size": {32: 1, 64: 2, 128: 3},
                 "learning_rate": {0.01: 1, 0.001: 2, 0.0001: 3}})
df['learning_rate'] = df['learning_rate'].astype(np.int64)

df2 = df.groupby(["time_level", "layers", "nodes", "cores", "batch_size", "learning_rate"]).agg({"accuracy": "mean"})
df2.columns = ['accuracy_mean']
df2 = df2.reset_index()
print(df)
print(df2)

train, test = train_test_split(df2, test_size=0.2)
train = train.reset_index().drop(columns='index')
test = test.reset_index().drop(columns='index')
print(train)
print(test)

df.to_csv("D:/logging/results/data_preprocessed.csv", index=False)
train.to_csv("D:/logging/results/data_train.csv", index=False)
test.to_csv("D:/logging/results/data_test.csv", index=False)

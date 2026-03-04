import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

n_estimators = params["model"]["n_estimators"]

train_data = pd.read_csv("/data2/ml-pipeline/data/fe/train_fe.csv")
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

gb = GradientBoostingClassifier(n_estimators=n_estimators)

gb.fit(X_train, y_train)

pickle.dump(gb, open("model.pkl", "wb"))

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score


test_data = pd.read_csv("/data2/ml-pipeline/data/fe/test_fe.csv")

clf = pickle.load(open("model.pkl", "rb"))

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)


model_evals = {
    "Accuracy": f"{accuracy_score(y_test, y_pred)}",
    "Precision": f"{precision_score(y_test, y_pred)}",
    # "ROC AUC Score": f"{roc_auc_score(y_test, y_prob)}",
    "Recall": f"{recall_score(y_test, y_pred)}"

}

with open("metrics.json", "w") as f:
    json.dump(model_evals, f)


 
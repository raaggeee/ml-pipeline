import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import logging

logger = logging.getLogger("Model Training")
logger.setLevel("DEBUG")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

console_logger = logging.StreamHandler()
console_logger.setFormatter(formatter)
console_logger.setLevel("DEBUG")
logger.addHandler(console_logger)

file_logger = logging.FileHandler("model_train.log")
file_logger.setFormatter(formatter)
file_logger.setLevel("ERROR")
logger.addHandler(file_logger)

def load_df(path):
    df = pd.read_csv(path)
    logging.info("Opened DF...")
    return df

def load_pkl(pkl_path):
    clf = pickle.load(open(pkl_path, "rb"))
    return clf

def eval_model(test_data, clf):
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

    return model_evals

def save_json(path, model_evals):
    with open(path, "w") as f:
        json.dump(model_evals, f)

def main():
    test_data = load_df("/data2/ml-pipeline/data/fe/test_fe.csv")
    clf = load_pkl("model.pkl")

    result = eval_model(test_data, clf)

    save_json("metrics.json", result)

if __name__ == "__main__":
    main()

 
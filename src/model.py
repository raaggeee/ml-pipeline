import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml
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

def load_yaml(path):
    with open(path, "r") as f:
        params = yaml.safe_load(f)

    logging.info("Opened YAML File...")

    return params

def load_df(path):
    df = pd.read_csv(path)
    logging.info("Opened DF...")
    return df

def train_model(train_data, n_estimators):
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    gb = GradientBoostingClassifier(n_estimators=n_estimators)

    gb.fit(X_train, y_train)

    pickle.dump(gb, open("model.pkl", "wb"))

def main():
    params = load_yaml("params.yaml")
    n_estimators = params["model"]["n_estimators"]

    train_df = load_df("/data2/ml-pipeline/data/fe/train_fe.csv")

    train_model(train_df, n_estimators)

if __name__ == "__main__":
    main()

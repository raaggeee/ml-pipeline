import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import logging

logger = logging.getLogger("Feature Engineering")
logger.setLevel("DEBUG")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

console_logger = logging.StreamHandler()
console_logger.setFormatter(formatter)
console_logger.setLevel("DEBUG")
logger.addHandler(console_logger)

file_logger = logging.FileHandler("feature_engineering.log")
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

def save_df(df, df_name):
    os.makedirs("/data2/ml-pipeline/data/fe", exist_ok=True)
    os.chdir("/data2/ml-pipeline/data/fe")
    df.to_csv(df_name)
    logging.info("Saved modified data...")

def normalize_features(df, n_features):
    X = df["comment"]
    y = df["label"]

    vectorizer = CountVectorizer(max_features=n_features)

    X_bow = vectorizer.fit_transform(X)

    df = pd.DataFrame(X_bow.toarray())
    df["label"] = y

    return df

def main():
    params = load_yaml("params.yaml")
    n_features = params["feature_engineering"]["n_festures"]

    train_data = load_df("/data2/ml-pipeline/data/preprocessed/train_preprocessed.csv")
    test_data = load_df("/data2/ml-pipeline/data/preprocessed/test_preprocessed.csv")

    train_df = normalize_features(train_data, n_features)
    test_df = normalize_features(test_data, n_features)

    save_df(train_df, "train_fe.csv")
    save_df(test_df, "test_fe.csv")


if __name__ == "__main__":
    main()

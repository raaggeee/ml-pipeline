import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

logger = logging.getLogger("Data Ingestion")
logger.setLevel("DEBUG")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

console_logger = logging.StreamHandler()
console_logger.setFormatter(formatter)
console_logger.setLevel("DEBUG")
logger.addHandler(console_logger)

file_logger = logging.FileHandler("data_ingestion.log")
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
    os.makedirs("/data2/ml-pipeline/data/raw", exist_ok=True)
    os.chdir("/data2/ml-pipeline/data/raw")
    df.to_csv(df_name)
    logging.info("Saved modified data...")

def modify_df(df):
    df.columns = ["tweet", "comment", "label"]

    df.drop("tweet", axis=1, inplace=True)
    logging.info("Dropped Tweet column")

    final_df = df[(df["label"] == "happy") | (df["label"] == "sad")]
    final_df["label"] = final_df["label"].replace({"happy": 0, "sad": 1})
    logging.info("Modified data")

    return final_df

def main():
    params = load_yaml("params.yaml")
    test_size = params["data_ingestion"]["test_size"]
    
    df_path = "/data2/ml-pipeline/data/smile-annotations-final.csv"
    df = load_df(df_path)

    final_df = modify_df(df)
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
    save_df(train_data, "train.csv")
    save_df(test_data, "test.csv")

if __name__ == "__main__":
    main()





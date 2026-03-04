import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

test_size = params["data_ingestion"]["test_size"]

df = pd.read_csv("/data2/ml-pipeline/data/smile-annotations-final.csv")
df.columns = ["tweet", "comment", "label"]

df.drop("tweet", axis=1, inplace=True)


final_df = df[(df["label"] == "happy") | (df["label"] == "sad")]
final_df["label"] = final_df["label"].replace({"happy": 0, "sad": 1})
train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

data_path = os.path.join("data", "raw")
os.makedirs(data_path, exist_ok=True)
os.chdir(data_path)
train_data.to_csv("train.csv")
test_data.to_csv("test.csv")


import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer

train_data = pd.read_csv("/data2/ml-pipeline/data/preprocessed/train_preprocessed.csv")
test_data = pd.read_csv("/data2/ml-pipeline/data/preprocessed/test_preprocessed.csv")

X_train = train_data["comment"]
y_train = train_data["label"]

X_test = test_data["comment"]
y_test = test_data["label"]

vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

train_df = pd.DataFrame(X_train_bow.toarray())
train_df["label"] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())
test_df["label"] = y_test

os.makedirs("data/fe", exist_ok=True)
os.chdir("data/fe")
train_df.to_csv("train_fe.csv")
test_df.to_csv("test_fe.csv")



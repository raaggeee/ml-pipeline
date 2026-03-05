import os
import pandas as pd
import nltk
import numpy as np
import re 
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download("wordnet")
nltk.download("stopwords")

def load_df(df_path):
    data = pd.read_csv(df_path)
    return data

def lemmatization(text):
    wordnet = WordNetLemmatizer()
    
    text = text.split()

    text = [wordnet.lemmatize(y) for y in text]

    return " ".join(text)

def stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(text)

def remove_numbers(text):
    remove_num = [i for i in text if not i.isdigit()]
    return "".join(remove_num)

def normalize_text(df):
    df.comment = df.comment.apply(lambda comment : lemmatization(comment))
    df.comment = df.comment.apply(lambda comment : stop_words(comment))
    df.comment = df.comment.apply(lambda comment : remove_numbers(comment))

    return df

def save_df(df, df_name):
    os.makedirs("/data2/ml-pipeline/data/preprocessed", exist_ok=True)
    os.chdir("/data2/ml-pipeline/data/preprocessed")
    df.to_csv(df_name)

def main():
    raw_data = os.path.join("data", "raw")
    os.chdir(raw_data)

    train_data = load_df("train.csv")
    test_data = load_df("test.csv")

    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)

    save_df(train_processed_data, "train_preprocessed.csv")
    save_df(test_processed_data, "test_preprocessed.csv")

if __name__ == "__main__":
    main()



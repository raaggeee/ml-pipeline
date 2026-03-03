import os
import pandas as pd
import nltk
import numpy as np
import re 
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

raw_data = os.path.join("data", "raw")
os.chdir(raw_data)

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

nltk.download("wordnet")
nltk.download("stopwords")

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

train_processed_data = normalize_text(train_data)
test_processed_data = normalize_text(test_data)

#save dir
save_data = os.path.join("data", "preprocessed")
os.chdir("../..")
os.makedirs(save_data, exist_ok=True)
os.chdir(save_data)

train_processed_data.to_csv("train_preprocessed.csv")
test_processed_data.to_csv("test_preprocessed.csv")




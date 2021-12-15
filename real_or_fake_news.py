from numpy.lib.function_base import extract
import pandas as pd
import re
import nltk
from scipy.sparse import data
nltk.download("wordnet")
from textblob import Word
import textfeatures as tf
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from sklearn import preprocessing

CLASSIFIER_MODEL = joblib.load("model.pkl")

#Open and read the user's text file
f = open("article.txt", "r")
file_lines = f.readline()
input_title = file_lines[0]
input_text = file_lines[2:]
input_date = file_lines[1]
print(input_text)
#Create a dataset from the input
dataset = [[input_title, input_text]]
dataset = pd.DataFrame(dataset, columns=['title', 'text'])

def extract_data(dataset):

    #Extracting amount of capitalized/lowercase words in the title
    print("Extracting numerical text data...")
    dataset["titleUpperCaseCount"] = dataset['title'].str.findall(r'[A-Z]').str.len()
    dataset["titleLowerCaseCount"] = dataset['title'].str.findall(r'[a-z]').str.len()
    #Extracting amount of capitalized/lowercase words in the text
    dataset["textUpperCaseCount"] = dataset['text'].str.findall(r'[A-Z]').str.len()
    dataset["textLowerCaseCount"] = dataset['text'].str.findall(r'[a-z]').str.len()

    #Extract the month  from the texts, then drop 'date'
    print("Extracting date data....")
    dataset["Month"]= dataset["date"].str.split(" ").str[0]
    dataset = dataset.drop(["date"], axis=1)

    #Encode Subject Label & month
    print("Encoding data....")
    label_encoder = preprocessing.LabelEncoder()
    dataset["subject"] = label_encoder.fit_transform(dataset['subject'])
    dataset["Month"] = label_encoder.fit_transform(dataset['Month'])

    #Extract feature counts from words
    print("Removing stop words....")
    tf.word_count(dataset,"text", "word_count")
    tf.stopwords_count(dataset, "text", "stopword_count")

    #Get words counts(in groups of 1 to 2)
    print("Conducting Count Vectorizer....")
    vectorizer = CountVectorizer(ngram_range=(1,2), max_features=1000, dtype=float)
    counts_sparse = vectorizer.fit_transform(dataset['text'])
    counts = pd.DataFrame(counts_sparse.toarray(), index=dataset.index, columns=vectorizer.get_feature_names_out())

    #Concat dataframes
    print("Concatenating dataframes....")
    dataset = pd.concat([dataset, counts], axis=1)
     
    return dataset

CLASSIFIER_MODEL.predict(extract_data(dataset))
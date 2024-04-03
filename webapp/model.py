import pandas as pd
import math
from collections import Counter
import numpy as np
import pickle
from sklearn import metrics
import re

df = pd.read_csv("./data/dataset.csv")
df = df.loc[:5000]
df.shape
first_col = df.iloc[1:, 0]
second_col = df.iloc[1:, 1]
second_col = second_col.fillna(0)
stop_words_file = open("./data/stopwords.txt", "r", encoding="utf-8")
stop_words = stop_words_file.read()
stop_words = stop_words.split("\n")


# data cleaning method
def data_cleaning(string):
    text = re.sub(
        "\,|\@|\-|\"|'| \)|\(|\)| \{| \}| \[| \]|!|‘|’|“|”| \:-|\?|।|/|\—|\०|\१|\२|\३|\४|\५|\६|\७|\८|\९|[0-9]",
        "",
        string,
    )
    return text


def stop_word_remove(array_element):
    array_element_set = set(array_element)
    final_list = list(array_element_set.difference(stop_words))
    return final_list


def tokenize():
    data_with_split = []
    for data in first_col:
        return_string = data_cleaning(data)
        each_docs = return_string.split()
        string_after_remove_word = stop_word_remove(each_docs)
        # print(string_after_remove_word)
        data_with_split.append(string_after_remove_word)
    return data_with_split  # it returns arr of each docs with spleted words


corpus = tokenize()


class TFIDFVectorizer:
    def __init__(self):
        self.vocabulary = None
        self.idf = None
        self.vocabulary = set()
        # Build vocabulary
        for document in corpus:
            self.vocabulary.update(document)
        self.vocabulary = list(self.vocabulary)

    def fit_transform(self, corpus):
        # Calculate IDF
        idf = {}
        N = len(corpus)
        for term in self.vocabulary:
            df = sum(1 for document in corpus if term in document)
            idf[term] = math.log(N / (1 + df))

        # Transform documents to TF-IDF representation
        tfidf_matrix = np.zeros((len(corpus), len(self.vocabulary)))
        for i, document in enumerate(corpus):
            tf = Counter(document)
            total_terms = len(document)
            for j, term in enumerate(self.vocabulary):
                if total_terms != 0:
                    tfidf_matrix[i, j] = (tf.get(term, 0) / total_terms) * idf[term]
                else:
                    tfidf_matrix[i, j] = 0  # Set TF-IDF to 0 if total_terms is 0

        self.idf = idf
        return tfidf_matrix

    def transform(self, corpus):
        tfidf_matrix = np.zeros((len(corpus), len(self.vocabulary)))
        for i, document in enumerate(corpus):
            tf = Counter(document)
            total_terms = len(document)
            for j, term in enumerate(self.vocabulary):
                if total_terms != 0:
                    tfidf_matrix[i, j] = (tf.get(term, 0) / total_terms) * self.idf.get(
                        term, 0
                    )
                else:
                    tfidf_matrix[i, j] = 0  # Set TF-IDF to 0 if total_terms is 0
        return tfidf_matrix


# Create TFIDFVectorizer instance
tfidf_vectorizer = TFIDFVectorizer()

# Fit and transform corpus
features = tfidf_vectorizer.fit_transform(corpus)

vectorizer_data = open("./webapp/model/vectorizer.pkl", "wb")
pickle.dump(tfidf_vectorizer, vectorizer_data)
vectorizer_data.close()

with open("./webapp/model/vectorizer.pkl", "rb") as tfidf:
    vectorizer = pickle.load(tfidf)
from sklearn.model_selection import train_test_split

x = features
y = second_col
train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.2, random_state=100
)

from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()
TrainData = naive_bayes.fit(train_x, train_y)

classifier_data = open("./webapp/model/classifier.pkl", "wb")
pickle.dump(naive_bayes, classifier_data)
classifier_data.close()
with open("./webapp/model/classifier.pkl", "rb") as pickle_saved_data:
    unpickled_data = pickle.load(pickle_saved_data)


prediction = unpickled_data.predict(test_x)


def calculate_performance_metrics(true_labels, predicted_labels):
    precision = metrics.precision_score(
        true_labels, predicted_labels, average="weighted"
    )
    recall = metrics.recall_score(true_labels, predicted_labels, average="weighted")
    accuracy = metrics.accuracy_score(true_labels, predicted_labels)
    f1_score = metrics.f1_score(true_labels, predicted_labels, average="weighted")

    return precision, recall, accuracy, f1_score


# Example usage:
precision, recall, accuracy, f1_score = calculate_performance_metrics(
    test_y, prediction
)

print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("F1 Score:", f1_score)


def predict_sentiment(sentence):
    # Preprocess the input sentence
    cleaned_sentence = data_cleaning(sentence)
    tokenized_sentence = stop_word_remove(cleaned_sentence.split())

    # Transform the preprocessed sentence using TF-IDF vectorizer
    sentence_features = vectorizer.transform([tokenized_sentence])

    # Use the trained classifier to predict the sentiment label
    predicted_label = unpickled_data.predict(sentence_features)

    return predicted_label[0]  # Return the predicted sentiment label


# Example usage:
sentence = "म नराम्रो केटा हुँ।"
predicted_sentiment = predict_sentiment(sentence)
print("Predicted Sentiment Label:", predicted_sentiment)

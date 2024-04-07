import math
import numpy as np
from collections import Counter


class TFIDFVectorizer:
    def __init__(self):
        self.vocabulary = None
        self.idf = None

    def get_tf_idf_info(self,words,sentence_feature):
        tf_idf_info = {}
        for word in words:
            if word not in self.vocabulary:
                tf_idf_info[word]=0
            else:
                index = self.vocabulary.index(word)
                tf_idf_info[word]=sentence_feature[0][index]
        return tf_idf_info
    
    def fit_transform(self, corpus):
        self.vocabulary = set()
        # Build vocabulary
        for document in corpus:
            self.vocabulary.update(document)
        self.vocabulary = list(self.vocabulary)
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

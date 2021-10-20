import eda

import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def get_doc_word_matrix(corpus):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(corpus)

"""
Decide on number of topics for LDA by considering the perplexity.
Divide into training and validation data.
"""
def choose_num_topics(documents):
    pass

"""
Run LDA with the text data and num of topics.
"""
def find_topics(documents, num_topics):
    pass

def main():
    df = eda.import_data("prepared_data.csv")
    documents = df["preparedText"] 
    print(documents.head())

if __name__ == "__main__":
    main()

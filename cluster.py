import eda

import pandas as pd
import numpy as np
import ast

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def get_doc_word_matrix(corpus):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(corpus)

"""
Decide on hyperparameters for LDA model.
"""
def choose_hyperparameters(X):
    # X_train, X_val, y_train, y_val = train_test_split(X, test_size=0.2)
    search_params = {'n_components': [5, 10, 15, 20]}

    lda = LatentDirichletAllocation(max_iter=5, learning_method='online')
    model = GridSearchCV(lda, param_grid=search_params)

    model.fit(X)
    return model
    

"""
Run LDA with the text data and num of topics.
"""
def find_topics(documents, num_topics):
    pass

def main():
    df = eda.import_data("prepared_data.csv")
    documents = df["preparedText"].tolist() 
    documents = [" ".join(ast.literal_eval(document)) for document in documents]
    X = get_doc_word_matrix(documents)

if __name__ == "__main__":
    main()

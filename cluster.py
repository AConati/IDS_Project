import eda

import pandas as pd
import numpy as np
import ast

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

"""
When the token lists (eg. ["This", "is", "a", "review"])
are written and loaded from a file, they are read as strings.
This converts them back to list form, and then back to
string form for use by the sklearn NLP models (ie. "This is a review").

Returns df where "preparedText" column is modified to reflect this conversion,
and the raw list of converted documents itself.
"""
def convert_loaded_text(df):
    documents = df["preparedText"].tolist()
    documents = [" ".join(ast.literal_eval(document)) for document in documents]
    df["preparedText"] = pd.Series(documents)
    return df, documents

def get_doc_word_matrix(corpus):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(corpus)

"""
Decide on hyperparameters for LDA model.
Note that this function is slow since it needs to consider
several combinations of hyperparameters.
"""
def choose_hyperparameters(X):
    # X_train, X_val, y_train, y_val = train_test_split(X, test_size=0.2)
    search_params = {'n_components': [5, 10, 15, 20], 'learning_decay': [.5, .7, .9]}

    lda = LatentDirichletAllocation(max_iter=5, learning_method='online')
    model = GridSearchCV(lda, param_grid=search_params)

    model.fit(X)
    return model
    

"""
Run LDA with the given documents.
"""
def find_topics(model, documents):
    pass

def main():
    df = eda.import_data("prepared_data.csv")
    df, documents = convert_loaded_text(df)
    X = get_doc_word_matrix(documents)

    grid_model = choose_hyperparameters(X)
    print(grid_model.best_params_)
    print(grid_model.best_score_)
    lda_model = grid_model.best_estimator_

if __name__ == "__main__":
    main()

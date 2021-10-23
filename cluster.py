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
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

"""
Decide on hyperparameters for LDA model.
Note that this function is slow since it needs to consider
several combinations of hyperparameters.
"""
def choose_hyperparameters(X):
    search_params = {'n_components': [5, 10, 15, 20], 'learning_decay': [.5, .7, .9]}

    lda = LatentDirichletAllocation(max_iter=5, learning_method='online')
    model = GridSearchCV(lda, param_grid=search_params)

    model.fit(X)
    return model
    

"""
Run LDA with the given documents.
"""
def find_topics(model, X):
    topics = model.transform(X)
    topic_names = ["Topic" + str(i) for i in range(model.n_components)]
    df = pd.DataFrame(topics, columns=topic_names)

    df["mostRelatedTopic"] = np.argmax(df.values, axis=1)

    return df

def print_top_words(lda_model, vocab, n_top_words):
    # topic_words = {}
    for topic, comp in enumerate(lda_model.components_):
        print('Topic{}'.format(topic))
        print(''.join([vocab[i] + ' ' + str(round(comp[i], 2)) + ' | '
            for i in comp.argsort()[:-n_top_words - 1:-1]]))
        """
        word_idx = np.argsort(comp)[::-1][:n_top_words]
        topic_words[topic] = [vocab[i] for i in word_idx]

    for topic, words in topic_words.items():
        print("Topic: {}".format(topic))
        print(" {}".format(" ".join(words)))
        """

def main():
    df = eda.import_data("prepared_data.csv")
    df, documents = convert_loaded_text(df)
    vectorizer, X = get_doc_word_matrix(documents)
    # X_train, X_test = train_test_split(X, test_size=0.2)

    """
    Tuning hyperparameters is slow, so commented out
    after running once. Should do again if data changes.
    """

    # print("Tuning hyperparameters...")
    # grid_model = choose_hyperparameters(X)
    # lda_model = grid_model.best_estimator_

    lda_model = LatentDirichletAllocation(max_iter=5, learning_method='online', n_components=5,
            learning_decay=.9)
    lda_model.fit(X)

    topic_df = find_topics(lda_model, X)
    print(topic_df.head())

    vocab = vectorizer.get_feature_names()
    print_top_words(lda_model, vocab, 20)

if __name__ == "__main__":
    main()

import parse_data

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def import_data(path):
    return pd.read_csv(path)

def get_idf_scores(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    df = pd.DataFrame(vectorizer.idf_, index=vectorizer.get_feature_names())
    df.columns = ['idf']
    return df.sort_values('idf')


def main():
    df = import_data("prepared_data.csv")
    pos = df[df['stars'] > 3]
    neg = df[df['stars'] < 3]
    pos_corpus = pos["preparedText"]
    neg_corpus = neg["preparedText"]
    pos_idf_scores = get_idf_scores(pos_corpus)
    neg_idf_scores = get_idf_scores(neg_corpus)
    print(pos_idf_scores.head())
    print(neg_idf_scores.head())

if __name__ == "__main__":
    main()

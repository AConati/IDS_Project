import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import json
from tqdm import tqdm

import time

def read_data(path):
    return pd.read_json(path, lines=True)


def prepare_text_data(review_texts, method="stem"):
    prepared_reviews = []
    for review_text in tqdm(review_texts):
        review_text = review_text.lower()
        tokens = nltk.word_tokenize(review_text)
        tokens = [token for token in tokens if token not in stopwords.words('english')]
        if method == "stem":
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(token) for token in tokens]
        else:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        prepared_reviews.append(tokens)

    return prepared_reviews


def get_df(review_path, business_path):
    review = read_data(review_path)

    business = read_data(business_path)
    business = business.drop(columns=["stars"])

    # Combine review data with business information
    df = pd.merge(review, business, on="business_id")
    df = df.drop(columns=["review_id", "business_id", "user_id", "address", "hours"])     

    # Split business attributes column
    # Note: apply is hella slow - pd.json_normalize method is much faster and seems
    # to be otherwise equivalent

    # attributes = df["attributes"].apply(pd.Series)
    attributes = pd.json_normalize(df["attributes"])
    df = pd.concat([df.drop(columns=["attributes"]), attributes], axis=1)

    # extract year from date, exact date probably not important
    df["year"] = df["date"].dt.year
    df = df.drop(columns=["date"])

    # Prepare text data for NLP
    # This takes a long time for large data sets
    # Do this once, then write the prepared data to skip
    # this step on subsequent calls
    start = time.time()
    print("Started text preparation...")
    prepared_reviews = prepare_text_data(df["text"].tolist())
    print("Preparing text took {.2f} seconds".format(time.time()-start))
    df["preparedText"] = pd.Series(prepared_reviews)

    return df


def main():
    df = get_df("review_subset1.json", "RAW_DATA/yelp_academic_dataset_business.json")
    print(df.head())
    print(df.shape)
    print(df.columns)
    print(df["year"].head())

    # After preparing data, we can write it to file so loading
    # is much faster on subsequent calls
    df.to_csv('prepared_data.csv')

if __name__ == "__main__":
    main()

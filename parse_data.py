import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import json
from tqdm import tqdm

import sys
import argparse

def read_data(path, nrows=None):
        return pd.read_json(path, lines=True, nrows=nrows)


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

"""
Prepared text data stored in column "preparedText" as
a list of tokens ("words").
Nrows parameter indicates how many lines to read.
Business id if included indicates which restaurants reviews to include from data provided.
"""

def get_df(review_path, business_path, nrows=None, business_name=None):
    print("Reading review data...")
    review = read_data(review_path, nrows=nrows)

    business = read_data(business_path)
    business = business.drop(columns=["stars"])

    # Combine review data with business information
    df = pd.merge(review, business, on="business_id")


    if business_name != None:
        index = df[df["name"] == business_name].index[0]
        business_id = df.iloc[[index]].iloc[0]["business_id"]
        df = df.drop(df[df["business_id"] != business_id].index)
        df["index"] = [x for x in range(len(df))]
        df = df.set_index("index")

    df = df.drop(columns=["review_id", "business_id", "user_id", "address", "hours"])     

    # Split business attributes column
    # Note: apply is pretty slow - pd.json_normalize method is much faster and seems
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
    print("Preprocessing review texts...")
    prepared_reviews = prepare_text_data(df["text"].tolist())
    df["preparedText"] = pd.Series(prepared_reviews)

    return df


def main():
    """
    Allows for reading restaurant name as command line argument.
    eg. python parse_data.py Oskar\ Blues\ Taproom

    Just trying to read the entire dataset made my computer tosi angry...
    a bit of a problem since a given restaurant's reviews are scattered
    across the entire dataset. 
    """
    if len(sys.argv) > 1:
        business_name = sys.argv[1]
        path = "RAW_DATA/yelp_academic_dataset_review.json"
        out = "{}_reviews".format(business_name)
        nrows = 500000
    else:
        business_name = None
        path = "RAW_DATA/small_review_subset.json"
        out = "prepared_data_small.csv"
        nrows = None

    df = get_df(path, "RAW_DATA/yelp_academic_dataset_business.json", business_name=business_name, nrows=nrows)
    print(df.head())
    print(df.shape)
    print(df.columns)
    print(df["year"].head())

    # After preparing data, we can write it to file so loading
    # is much faster on subsequent calls
    df.to_csv(out)

if __name__ == "__main__":
    main()

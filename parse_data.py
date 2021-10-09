import pandas as pd
import json

def read_data(path):
    return pd.read_json(path, lines=True)

def get_df(review_path, business_path):
    review = read_data(review_path)
    business = read_data(business_path)
    business = business.drop(columns=["stars"])
    df = pd.merge(review, business, on="business_id")
    df = df.drop(columns=["review_id", "business_id", "user_id", "address"])
    return df


def main():
    df = get_df("review_subset1.json", "RAW_DATA/yelp_academic_dataset_business.json")
    print(df["attributes"].head())
    print(df.shape)
    print(df.columns)

if __name__ == "__main__":
    main()

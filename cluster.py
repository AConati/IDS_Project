import pandas as pd
import numpy as np
import ast

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import sys

def convert_loaded_text(df):
    """
    When the token lists (eg. ["This", "is", "a", "review"])
    are written and loaded from a file, they are read as strings.
    This converts them back to list form, and then back to
    string form for use by the sklearn NLP models (ie. "This is a review").

    Returns df where "preparedText" column is modified to reflect this conversion,
    and the raw list of converted documents itself.
    """

    documents = df["preparedText"].tolist()
    documents = [" ".join(ast.literal_eval(document)) for document in documents]
    df["preparedText"] = pd.Series(documents)
    return df, documents

def get_doc_word_matrix(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

def choose_hyperparameters(X):
    """
    Decide on hyperparameters for LDA model.
    Note that this function is slow since it needs to consider
    several combinations of hyperparameters.
    """

    search_params = {'n_components': [5, 10, 15, 20], 'learning_decay': [.5, .7, .9]}

    lda = LatentDirichletAllocation(max_iter=5, learning_method='online')
    model = GridSearchCV(lda, param_grid=search_params)

    model.fit(X)
    return model
    

def find_topics(model, X):
    """
    Run LDA with the given documents.
    """

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

def show_topic_word_cloud(df, topic_number, ignore=[]):
    """
    Shows word cloud for reviews most related to given topic number.
    Works kind of stupidly at the moment because it doesn't factor in
    HOW related it is.
    """

    df = df[df["mostRelatedTopic"] == topic_number]
    texts = df["preparedText"]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    idx = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names()

    freqs = {word: X.getcol(idx).sum() for word, idx in vectorizer.vocabulary_.items() if word not in ignore}
    cloud = WordCloud(width=800, height=600, mode='RGBA', background_color='white', max_words=100).fit_words(freqs)
    plt.imshow(cloud)
    plt.show()

def get_topic_reviews(df, num_topics):
    """
    Gets number of reviews by topic (most related).
    """
    relevance = {}
    for x in range(num_topics):
        col_name = "Topic{}".format(x)
        df = df.sort_values(col_name, ascending=False)
        relevance[col_name] = df[col_name].head()
        df["text"].head().to_csv("{}.csv".format(col_name))

    return relevance, df["mostRelatedTopic"].value_counts()

def get_weighted_averages(df, num_topics):
    """
    Gets weighted average score across different topics.
    """

    topic_q = {"Topic{}".format(i): df["Topic{}".format(i)].sum() for i in range(num_topics)}
    topic_wa = {"Topic{}".format(i): (df["stars"]*df["Topic{}".format(i)]).sum()/topic_q["Topic{}".format(i)] for i in range(num_topics)}
    return topic_wa

def weighted_averages_by_year(df, num_topics):
    years = df["year"].unique()
    year_dfs = [df[df['year'] == years[i]] for i in range(len(years))]
    topic_dfs = [[] for i in range(num_topics)]
    for i, year_df in enumerate(year_dfs):
        topic_wa = get_weighted_averages(year_df, num_topics)
        for x in range(num_topics):
            topic_dfs[x].append((years[i], topic_wa["Topic{}".format(x)]))

    topic_dfs = [pd.DataFrame(x, columns=['year', 'score']) for x in topic_dfs]
    topic_dfs = [x.sort_values(["year"]) for x in topic_dfs]
    return topic_dfs


def topicnize_new_reviews(model, vectorizer, path):
    df = pd.read_csv(path)
    df, documents = convert_loaded_text(df)
    X = vectorizer.transform(documents)
    topic_probabilities = model.transform(X)
    topic_df = pd.DataFrame(topic_probabilities, columns=["Topic{}".format(i) for i in range(model.n_components)])
    topic_df = pd.concat([df, topic_df], axis=1)
    path_wo_ext = path[0: path.index(".")]
    topic_df.to_csv("{}_topicnized.csv".format(path_wo_ext))
    return topic_df

def main():
    print("Parsing inputs...")
    df = pd.read_csv("prepared_data.csv")
    df, documents = convert_loaded_text(df)
    vectorizer, X = get_doc_word_matrix(documents)
    # X_train, X_test = train_test_split(X, test_size=0.2)

    # Tuning hyperparameters is slow, so commented out
    # after running once. Should do again if data changes.

    print("Tuning hyperparameters...")
    # grid_model = choose_hyperparameters(X)
    # lda_model = grid_model.best_estimator_
    # with open("lda_model_params.txt", "w") as f:
        # f.write(str(grid_model.best_params_))

    print("Fitting model...")
    lda_model = LatentDirichletAllocation(max_iter=5, learning_method='online', n_components=5, learning_decay=.9)
    lda_model.fit(X)

    topic_df = find_topics(lda_model, X)
    print(topic_df.head())

    vocab = vectorizer.get_feature_names()
    print_top_words(lda_model, vocab, 20)

    df = pd.concat([df, topic_df], axis=1)
    relevance, counts = get_topic_reviews(df, 5)
    topicnize_new_reviews(lda_model, vectorizer, "Burger_King_reviews.csv")
    topicnize_new_reviews(lda_model, vectorizer, "Legal_Sea_Foods_Reviews.csv")
    topicnize_new_reviews(lda_model, vectorizer, "White_Castle_reviews.csv")
    topicnize_new_reviews(lda_model, vectorizer, "Subway_reviews.csv")
    topicnize_new_reviews(lda_model, vectorizer, "Panda_Express_reviews.csv")
    topicnize_new_reviews(lda_model, vectorizer, "Chipotle_Mexican_Grill_reviews.csv")
    df.to_csv("topicnized_data.csv")
    if len(sys.argv) > 1:
        show_topic_word_cloud(df, int(sys.argv[1]), ignore=["good", "great", "food", "place"])

    

if __name__ == "__main__":
    main()

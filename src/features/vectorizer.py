from sklearn.feature_extraction.text import TfidfVectorizer

def create_vectorizer():

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=500
    )

    return vectorizer
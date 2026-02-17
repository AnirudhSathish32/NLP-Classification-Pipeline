from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_vectorizer():
    """
    TF-IDF converts text into weighted numerical features.

    Why TF-IDF:
    - Highlights informative words.
    - Downweights extremely common words.

    Why n-grams:
    - Captures short phrases (e.g., "not good")
    - Improves context representation.
    """
    return TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=5
    )

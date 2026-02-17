import re
from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer for cleaning text inside sklearn Pipeline.

    Includes:
    - lowercasing
    - punctuation removal
    - extra whitespace removal
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned = []
        for text in X:
            text = text.lower()
            text = re.sub(r"[^\w\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            cleaned.append(text)
        return cleaned

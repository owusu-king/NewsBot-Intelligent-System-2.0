"""
feature_extractor.py
---------------------
TF-IDF vectorization and custom feature engineering (text length, sentiment
scores, etc.) built on the midterm's Module 3 (TF-IDF Analysis).
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
    """Wraps TF-IDF + numeric features into a single feature matrix."""

    def __init__(self, max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )
        self.is_fitted = False

    def fit_transform(self, texts):
        matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return matrix

    def transform(self, texts):
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor must be fit before calling transform().")
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    def top_terms_for_category(self, tfidf_df, category_col, category, n_terms=10):
        """Average TF-IDF scores across all docs in a category -> top N terms."""
        category_data = tfidf_df[tfidf_df[category_col] == category]
        mean_scores = category_data.drop(columns=[category_col]).mean().sort_values(ascending=False)
        return mean_scores.head(n_terms)

    @staticmethod
    def text_length_features(texts):
        """Simple length-based numeric features: char count, word count, avg word length."""
        feats = []
        for t in texts:
            t = str(t)
            words = t.split()
            n_words = max(len(words), 1)
            feats.append([len(t), n_words, len(t) / n_words])
        return np.array(feats)

    @staticmethod
    def combine_features(*feature_arrays):
        """Horizontally stack TF-IDF (dense) + numeric feature arrays."""
        dense_arrays = []
        for arr in feature_arrays:
            if hasattr(arr, "toarray"):
                dense_arrays.append(arr.toarray())
            else:
                dense_arrays.append(np.asarray(arr))
        return np.hstack(dense_arrays)

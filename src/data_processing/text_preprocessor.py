"""
text_preprocessor.py
---------------------
Enhanced text preprocessing pipeline, built on the midterm's clean_text /
preprocess_text functions. Wrapped in a class so it can be reused across
every module of NewsBot 2.0 (classification, topic modeling, summarization,
multilingual analysis, etc.) with a single, consistent API.
"""
import re
import string
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


class TextPreprocessor:
    """Cleans and normalizes raw news text for downstream NLP tasks."""

    def __init__(self, language="english", remove_stopwords=True, lemmatize=True):
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer()
        try:
            news_stopwords = {"say", "mr", "mrs", "ms", "year", "new", "time", "people", "one", "would", "could", "also", "last", "first", "bn", "million", "billion", "said"}
            self.stop_words = set(stopwords.words(language))
            self.stop_words.update(news_stopwords)
        except OSError:
            self.stop_words = set()

    def clean_text(self, text: str) -> str:
        """Lowercase, strip HTML/URLs/special chars, collapse whitespace."""
        if pd.isna(text) or text is None:
            return ""
        text = str(text).lower()
        text = re.sub(r"<[^>]+>", " ", text)                       # HTML tags
        text = re.sub(r"http\S+|www\.\S+", " ", text)               # URLs
        text = re.sub(r"\S+@\S+", " ", text)                        # emails
        text = re.sub(r"[^a-z\s]", " ", text)                       # non-letters
        text = re.sub(r"\s+", " ", text).strip()                    # whitespace
        return text

    def preprocess_text(self, text: str) -> str:
        """Full pipeline: clean -> tokenize -> stopword removal -> lemmatize."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return ""
        try:
            tokens = word_tokenize(cleaned)
        except LookupError:
            tokens = cleaned.split()

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if len(t) > 1]
        return " ".join(tokens)

    def preprocess_dataframe(self, df: pd.DataFrame, text_col="content", title_col=None):
        """Add *_clean and *_processed columns to a dataframe in place, return it."""
        df = df.copy()
        df[f"{text_col}_clean"] = df[text_col].apply(self.clean_text)
        df[f"{text_col}_processed"] = df[text_col].apply(self.preprocess_text)
        if title_col and title_col in df.columns:
            df[f"{title_col}_clean"] = df[title_col].apply(self.clean_text)
            df[f"{title_col}_processed"] = df[title_col].apply(self.preprocess_text)
            df["full_text"] = df[title_col].fillna("") + ". " + df[text_col].fillna("")
        else:
            df["full_text"] = df[text_col]
        return df

"""
cross_lingual_analyzer.py
---------------------------
Compares news coverage across languages: sentiment differences, topic
overlap, and volume of coverage per language - the "Cultural Context" and
"Cross-Language Analysis" requirements.
"""
import pandas as pd

from .language_detector import LanguageDetector
from .translator import Translator
from ..analysis.sentiment_analyzer import SentimentAnalyzer


class CrossLingualAnalyzer:
    def __init__(self):
        self.detector = LanguageDetector()
        self.translator = Translator()
        self.sentiment = SentimentAnalyzer()

    def analyze_corpus(self, df: pd.DataFrame, text_col="content") -> pd.DataFrame:
        """Detect language, translate to English, and score sentiment - all in one pass."""
        df = self.detector.detect_dataframe(df, text_col=text_col)
        df = self.translator.translate_dataframe(df, text_col=text_col, lang_col="language_code")
        df = self.sentiment.analyze_dataframe(df, text_col="translated_text")
        return df

    def coverage_by_language(self, analyzed_df: pd.DataFrame) -> pd.DataFrame:
        """How much coverage (volume + average sentiment) exists per language."""
        return analyzed_df.groupby("language_name").agg(
            n_articles=("translated_text", "count"),
            avg_sentiment=("sentiment_compound", "mean"),
        ).sort_values("n_articles", ascending=False)

    def compare_topic_across_languages(self, analyzed_df: pd.DataFrame, topic_keyword: str, text_col="translated_text"):
        """Filter articles mentioning a topic keyword (post-translation) and compare by language."""
        mask = analyzed_df[text_col].str.contains(topic_keyword, case=False, na=False)
        subset = analyzed_df[mask]
        if subset.empty:
            return pd.DataFrame()
        return subset.groupby("language_name").agg(
            mentions=(text_col, "count"),
            avg_sentiment=("sentiment_compound", "mean"),
        )

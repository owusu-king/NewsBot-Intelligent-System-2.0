"""
sentiment_analyzer.py
-----------------------
VADER-based sentiment analysis (builds on midterm Module 6), extended with
sentiment evolution tracking over time/category for the final project's
"Sentiment Evolution" requirement.
"""
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> dict:
        if not text or pd.isna(text):
            return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "label": "neutral"}
        scores = self.sia.polarity_scores(str(text))
        if scores["compound"] >= 0.05:
            label = "positive"
        elif scores["compound"] <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        scores["label"] = label
        return scores

    def analyze_dataframe(self, df: pd.DataFrame, text_col="content") -> pd.DataFrame:
        records = df[text_col].apply(self.analyze).tolist()
        sentiment_df = pd.DataFrame(records)
        sentiment_df.index = df.index
        return pd.concat([df, sentiment_df.add_prefix("sentiment_")], axis=1)

    def evolution_by_group(self, sentiment_df: pd.DataFrame, group_col: str, value_col="sentiment_compound"):
        """Average sentiment per group (e.g. per category, or per time bucket)."""
        return sentiment_df.groupby(group_col)[value_col].agg(["mean", "std", "min", "max", "count"])

    def evolution_over_time(self, sentiment_df: pd.DataFrame, date_col: str, freq="W", value_col="sentiment_compound"):
        """Track how average sentiment changes over time (requires a datetime column)."""
        ts = sentiment_df.copy()
        ts[date_col] = pd.to_datetime(ts[date_col])
        return ts.set_index(date_col)[value_col].resample(freq).mean()

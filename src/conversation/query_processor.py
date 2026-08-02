"""
query_processor.py
--------------------
Processes natural language queries against an analyzed news dataframe
(one that already has category, sentiment_label, sentiment_compound,
translated_text/content, and a timestamp column if time-filtering is used).

Follow-up context (e.g. "now show only the negative ones" reusing the prior
turn's category filter) is supported two ways:
  1. Implicitly, via self.context - convenient for single-user use in a
     notebook or script.
  2. Explicitly, by passing `prior_filters` into process() - this is what
     the Django app uses, storing filters in the per-user session, so two
     different visitors never share or overwrite each other's conversation
     state (self.context alone would be shared by every visitor, since the
     QueryProcessor instance lives once per server process).
"""
import pandas as pd

from .intent_classifier import IntentClassifier
from ..language_models.embeddings import EmbeddingIndex
from ..language_models.generator import InsightGenerator

RESULTS_PAGE_SIZE = 25


class QueryProcessor:
    def __init__(self, df: pd.DataFrame, text_col="content", date_col=None):
        self.df = df
        self.text_col = text_col
        self.date_col = date_col
        self.intent_classifier = IntentClassifier()
        self.generator = InsightGenerator()
        self.embedding_index = EmbeddingIndex()
        self.embedding_index.build(df[text_col].fillna("").tolist())
        self.context = {"last_filters": {}, "last_result_indices": None}

    def process(self, query: str, prior_filters: dict = None) -> dict:
        parsed = self.intent_classifier.classify(query)
        intent = parsed["intent"]
        entities = parsed["entities"]

        # Explicit "start over" / "clear filters" support
        if intent == "reset":
            merged_filters = {}
        else:
            base_filters = self.context["last_filters"] if prior_filters is None else prior_filters
            merged_filters = {**base_filters, **entities}

        results_df = self._apply_filters(merged_filters)
        response_text = self.generator.query_response_template(intent, results_df.to_dict("records"))

        # Only mutate shared internal state when the caller isn't managing
        # its own (session-scoped) filters explicitly.
        if prior_filters is None:
            self.context["last_filters"] = merged_filters
            self.context["last_result_indices"] = results_df.index.tolist()

        return {
            "query": query,
            "intent": intent,
            "filters_applied": merged_filters,
            "n_results": len(results_df),
            "results": results_df.head(RESULTS_PAGE_SIZE).to_dict("records"),
            "response": response_text,
        }

    def _apply_filters(self, filters: dict) -> pd.DataFrame:
        result = self.df

        if "category" in filters and "category" in result.columns:
            result = result[result["category"] == filters["category"]]

        if "sentiment" in filters and "sentiment_label" in result.columns:
            result = result[result["sentiment_label"] == filters["sentiment"]]

        if "time_window" in filters and self.date_col and self.date_col in result.columns:
            cutoff = pd.Timestamp.now() - pd.Timedelta(filters["time_window"])
            result = result[pd.to_datetime(result[self.date_col]) >= cutoff]

        return result

    def semantic_search(self, query: str, top_k: int = 5):
        """For free-text/topical queries not covered by structured filters."""
        hits = self.embedding_index.most_similar(query, top_k=top_k)
        for h in hits:
            h["article"] = self.df.iloc[h["index"]].to_dict()
        return hits

    def reset_context(self):
        self.context = {"last_filters": {}, "last_result_indices": None}

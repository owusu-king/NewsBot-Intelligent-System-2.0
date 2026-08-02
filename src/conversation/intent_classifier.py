"""
intent_classifier.py
----------------------
Rule/keyword-based intent detection for natural language queries like
"Show me positive tech news from this week". A lightweight, explainable,
fully offline classifier - appropriate given the small, fixed set of
intents this system needs to support (vs. training a full intent model).
"""
import re

INTENT_PATTERNS = {
    "reset": [r"\b(start over|clear filters|reset|new search|forget (that|it)|clear (that|it))\b"],
    "filter_by_category": [r"\b(tech|technology|sport|sports|business|politics|entertainment|health)\b"],
    "filter_by_sentiment": [r"\b(positive|negative|neutral)\b"],
    "filter_by_time": [r"\b(today|this week|this month|yesterday|last week|recent|latest)\b"],
    "count": [r"\bhow many\b", r"\bcount\b", r"\bnumber of\b"],
    "top_entities": [r"\b(who|which (people|organizations|companies))\b", r"\bmost mentioned\b", r"\btop entities\b"],
    "summarize": [r"\bsummar(y|ize)\b", r"\btl;?dr\b"],
    "compare": [r"\bcompare\b", r"\bversus\b", r"\bvs\.?\b"],
    "search": [r"\babout\b", r"\brelated to\b", r"\bmentioning\b"],
}

CATEGORY_WORDS = {
    "tech": "tech", "technology": "tech", "sport": "sport", "sports": "sport",
    "business": "business", "politics": "politics", "entertainment": "entertainment", "health": "health",
}
SENTIMENT_WORDS = ["positive", "negative", "neutral"]
TIME_WORDS = {
    "today": "1D", "yesterday": "1D", "this week": "7D", "last week": "7D",
    "this month": "30D", "recent": "7D", "latest": "7D",
}


class IntentClassifier:
    def classify(self, query: str) -> dict:
        query_lower = query.lower()
        matched_intents = []
        for intent, patterns in INTENT_PATTERNS.items():
            if any(re.search(p, query_lower) for p in patterns):
                matched_intents.append(intent)

        # "reset"/"clear filters" always takes priority over any other match
        if "reset" in matched_intents:
            primary_intent = "reset"
        # Combine category + sentiment filters into a single composite intent
        # when both are present (e.g. "positive tech news")
        elif "filter_by_category" in matched_intents and "filter_by_sentiment" in matched_intents:
            primary_intent = "filter_by_sentiment_category"
        elif matched_intents:
            primary_intent = matched_intents[0]
        else:
            primary_intent = "search"

        entities = self.extract_query_entities(query_lower)
        return {"intent": primary_intent, "all_matched_intents": matched_intents, "entities": entities}

    def extract_query_entities(self, query_lower: str) -> dict:
        entities = {}
        for word, cat in CATEGORY_WORDS.items():
            if word in query_lower:
                entities["category"] = cat
                break
        for word in SENTIMENT_WORDS:
            if word in query_lower:
                entities["sentiment"] = word
                break
        for phrase, window in TIME_WORDS.items():
            if phrase in query_lower:
                entities["time_window"] = window
                entities["time_phrase"] = phrase
                break
        return entities

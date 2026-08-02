"""
generator.py
-------------
Content enhancement and automatic insight generation. Combines
classification, sentiment, entities, and topics into human-readable
narrative insights ("Insight Generation" requirement) using template-based
natural language generation - fast, deterministic, and fully offline.
"""


class InsightGenerator:
    def enhance_article(self, article: dict) -> dict:
        """
        Given an analyzed article (category, sentiment, entities, topics, summary),
        produce a short contextual "enhancement" paragraph.
        """
        parts = []
        category = article.get("category")
        sentiment_label = article.get("sentiment_label", "neutral")
        entities = article.get("entities", [])
        confidence = article.get("confidence")

        if category:
            conf_text = f" (confidence: {confidence:.0%})" if confidence is not None else ""
            parts.append(f"This article was classified as **{category}**{conf_text}.")

        if sentiment_label:
            parts.append(f"Its overall tone is **{sentiment_label}**.")

        if entities:
            people = [e["text"] for e in entities if e["label"] == "PERSON"][:3]
            orgs = [e["text"] for e in entities if e["label"] == "ORG"][:3]
            places = [e["text"] for e in entities if e["label"] == "GPE"][:3]
            if people:
                parts.append(f"Key people mentioned: {', '.join(people)}.")
            if orgs:
                parts.append(f"Organizations involved: {', '.join(orgs)}.")
            if places:
                parts.append(f"Locations referenced: {', '.join(places)}.")

        return {"enhancement": " ".join(parts), **article}

    def generate_business_insights(self, analysis_summary: dict) -> list:
        """
        Turn aggregate stats (from utils.evaluation / classifier / sentiment)
        into a bulleted list of business-facing insights.
        """
        insights = []
        cat_counts = analysis_summary.get("category_counts", {})
        if cat_counts:
            top_cat = max(cat_counts, key=cat_counts.get)
            insights.append(f"'{top_cat}' is the most represented category, "
                             f"accounting for {cat_counts[top_cat]} of {sum(cat_counts.values())} articles.")

        sentiment_by_cat = analysis_summary.get("sentiment_by_category", {})
        if sentiment_by_cat:
            most_negative = min(sentiment_by_cat, key=sentiment_by_cat.get)
            most_positive = max(sentiment_by_cat, key=sentiment_by_cat.get)
            insights.append(f"'{most_positive}' coverage skews most positive "
                             f"(avg sentiment {sentiment_by_cat[most_positive]:.2f}), while "
                             f"'{most_negative}' skews most negative "
                             f"(avg sentiment {sentiment_by_cat[most_negative]:.2f}).")

        accuracy = analysis_summary.get("classifier_f1")
        if accuracy is not None:
            insights.append(f"The classification model achieves a weighted F1-score of {accuracy:.2f}, "
                             "suitable for automated content routing with human review on low-confidence cases.")

        top_entities = analysis_summary.get("top_entities")
        if top_entities:
            names = ", ".join(f"{name} ({count})" for name, count in top_entities[:5])
            insights.append(f"Most frequently mentioned entities: {names}.")

        return insights

    def query_response_template(self, intent: str, results: list) -> str:
        """Simple templated natural-language response for the conversational interface."""
        if intent == "reset":
            return f"Filters cleared. Showing all {len(results)} articles."
        if not results:
            return "I couldn't find any articles matching that request."
        n = len(results)
        if intent == "filter_by_sentiment_category":
            return f"I found {n} articles matching your filter. Here are the top results."
        if intent == "count":
            return f"There are {n} articles matching your query."
        return f"Here are {n} results for your query."

"""
response_generator.py
------------------------
Formats QueryProcessor output into a final, user-facing conversational
response (text + suggested follow-ups), used by both the notebooks and the
Django conversational interface view.
"""


class ResponseGenerator:
    def format_response(self, query_result: dict) -> dict:
        n = query_result["n_results"]
        intent = query_result["intent"]
        filters = query_result["filters_applied"]

        filter_desc = []
        if "category" in filters:
            filter_desc.append(f"category = {filters['category']}")
        if "sentiment" in filters:
            filter_desc.append(f"sentiment = {filters['sentiment']}")
        if "time_phrase" in filters:
            filter_desc.append(f"time = {filters['time_phrase']}")
        filter_text = "; ".join(filter_desc) if filter_desc else "no filters"

        message = query_result["response"]
        if n > 0:
            message += f" ({filter_text})"

        follow_ups = self._suggest_follow_ups(query_result)
        return {"message": message, "n_results": n, "intent": intent,
                "results_preview": query_result["results"][:15], "suggested_follow_ups": follow_ups}

    def _suggest_follow_ups(self, query_result):
        suggestions = []
        filters = query_result["filters_applied"]
        if "sentiment" not in filters:
            suggestions.append("show only the positive ones")
        if "category" not in filters:
            suggestions.append("just the tech articles")
        if query_result["intent"] != "summarize":
            suggestions.append("summarize these")
        if filters:
            suggestions.append("clear filters")
        return suggestions[:3]

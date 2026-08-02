from django.db import models


class AnalyzedArticle(models.Model):
    """A news article that has been run through the NewsBot NLP pipeline."""
    source_article_id = models.IntegerField(null=True, blank=True)
    title = models.CharField(max_length=500, blank=True)
    content = models.TextField()

    category = models.CharField(max_length=50, blank=True)
    category_confidence = models.FloatField(null=True, blank=True)

    sentiment_label = models.CharField(max_length=20, blank=True)
    sentiment_compound = models.FloatField(null=True, blank=True)

    language_code = models.CharField(max_length=10, blank=True, default="en")
    language_name = models.CharField(max_length=50, blank=True, default="English")
    translated_text = models.TextField(blank=True)

    summary = models.TextField(blank=True)
    entities_json = models.JSONField(default=list, blank=True)
    topic_id = models.IntegerField(null=True, blank=True)
    insight = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"[{self.category}] {self.title[:60]}"


class QueryLog(models.Model):
    """Logs natural-language queries made through the conversational interface."""
    query_text = models.CharField(max_length=500)
    detected_intent = models.CharField(max_length=50, blank=True)
    filters_applied = models.JSONField(default=dict, blank=True)
    n_results = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.query_text} ({self.n_results} results)"

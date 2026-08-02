from django.contrib import admin
from .models import AnalyzedArticle, QueryLog


@admin.register(AnalyzedArticle)
class AnalyzedArticleAdmin(admin.ModelAdmin):
    list_display = ("title", "category", "sentiment_label", "language_name", "created_at")
    list_filter = ("category", "sentiment_label", "language_name")
    search_fields = ("title", "content")


@admin.register(QueryLog)
class QueryLogAdmin(admin.ModelAdmin):
    list_display = ("query_text", "detected_intent", "n_results", "created_at")

from django.urls import path
from . import views

app_name = "newsbot_app"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("analyze/", views.analyze_article, name="analyze"),
    path("query/", views.query_interface, name="query"),
    path("batch/", views.batch_processing, name="batch"),
    path("api/analyze/", views.api_analyze, name="api_analyze"),
    path("api/query/", views.api_query, name="api_query"),
]

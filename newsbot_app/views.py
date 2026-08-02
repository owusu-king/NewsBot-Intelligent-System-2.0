import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .forms import ArticleAnalysisForm, QueryForm
from .nlp_service import NewsBotService
from .models import AnalyzedArticle, QueryLog


def dashboard(request):
    service = NewsBotService.get_instance()
    summary = service.dashboard_summary()
    topics = service.topics_overview()
    context = {
        "summary": summary,
        "topics": topics,
        "dataset_source": service.dataset_source,
        "model_trained_at": service.model_trained_at,
        "category_counts": summary.get("category_counts", {}),
        "sentiment_by_category": summary.get("sentiment_by_category", {}),
    }
    return render(request, "dashboard.html", context)


def analyze_article(request):
    service = NewsBotService.get_instance()
    result = None
    if request.method == "POST":
        form = ArticleAnalysisForm(request.POST)
        if form.is_valid():
            result = service.analyze_article(form.cleaned_data["title"], form.cleaned_data["content"])
            AnalyzedArticle.objects.create(
                title=form.cleaned_data["title"][:500],
                content=form.cleaned_data["content"],
                category=result["category"],
                category_confidence=result["confidence"],
                sentiment_label=result["sentiment_label"],
                sentiment_compound=result["sentiment_compound"],
                language_code=result["language_code"],
                language_name=result["language_name"],
                summary=result["summary"],
                entities_json=result["entities"],
                insight=result["enhancement"],
            )
    else:
        form = ArticleAnalysisForm()
    return render(request, "analyze.html", {"form": form, "result": result})


def query_interface(request):
    service = NewsBotService.get_instance()
    response = None
    history = QueryLog.objects.all()[:10]

    if request.method == "POST":
        form = QueryForm(request.POST)
        if form.is_valid():
            query_text = form.cleaned_data["query"]
            # Filters are stored in this user's session, NOT on the shared
            # NewsBotService singleton - otherwise every visitor to the site
            # would silently share (and stomp on) each other's filters.
            prior_filters = request.session.get("newsbot_filters", {})
            result = service.query_processor.process(query_text, prior_filters=prior_filters)
            request.session["newsbot_filters"] = result["filters_applied"]

            response = service.response_generator.format_response(result)
            QueryLog.objects.create(
                query_text=query_text, detected_intent=response["intent"],
                filters_applied=result["filters_applied"], n_results=response["n_results"],
            )
    else:
        form = QueryForm()

    active_filters = request.session.get("newsbot_filters", {})
    return render(request, "query.html", {
        "form": form, "response": response, "history": history, "active_filters": active_filters,
    })


def batch_processing(request):
    service = NewsBotService.get_instance()
    results = None
    if request.method == "POST" and request.FILES.get("csv_file"):
        import pandas as pd
        uploaded = request.FILES["csv_file"]
        df = pd.read_csv(uploaded)
        text_col = "content" if "content" in df.columns else df.columns[0]
        analyzed = []
        for _, row in df.head(50).iterrows():  # cap for demo responsiveness
            analyzed.append(service.analyze_article(row.get("title", ""), row[text_col]))
        results = analyzed
    return render(request, "batch.html", {"results": results})


# ---- JSON API endpoints (used by the bonus web app / external integrations) ----

@csrf_exempt
@require_http_methods(["POST"])
def api_analyze(request):
    service = NewsBotService.get_instance()
    data = json.loads(request.body or "{}")
    text = data.get("text", "")
    if not text:
        return JsonResponse({"error": "Missing 'text' field."}, status=400)
    result = service.analyze_article(data.get("title", ""), text)
    return JsonResponse(result)


@csrf_exempt
@require_http_methods(["POST"])
def api_query(request):
    """
    Stateless by design: the client (not the server) is responsible for
    conversation state. Pass back the `filters_applied` from a prior
    response as `context_filters` to continue that conversation thread;
    omit it (or send {}) to start fresh. This avoids sharing one visitor's
    filters with another over the JSON API.
    """
    service = NewsBotService.get_instance()
    data = json.loads(request.body or "{}")
    query_text = data.get("query", "")
    if not query_text:
        return JsonResponse({"error": "Missing 'query' field."}, status=400)
    prior_filters = data.get("context_filters", {})
    response = service.process_query(query_text, prior_filters=prior_filters)
    return JsonResponse(response)

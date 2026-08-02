import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.data_processing.data_loader import load_news_data
from src.data_processing.text_preprocessor import TextPreprocessor
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.multilingual.language_detector import LanguageDetector
from src.multilingual.translator import Translator
from src.multilingual.cross_lingual_analyzer import CrossLingualAnalyzer
from src.conversation.query_processor import QueryProcessor
from src.conversation.response_generator import ResponseGenerator
from src.conversation.intent_classifier import IntentClassifier
from src.language_models.summarizer import Summarizer
from src.language_models.embeddings import EmbeddingIndex
from src.language_models.generator import InsightGenerator
from src.utils.export import build_summary_report


def test_full_pipeline_runs_end_to_end():
    df = load_news_data()
    pre = TextPreprocessor()
    df = pre.preprocess_dataframe(df, text_col="content", title_col="title")

    sa = SentimentAnalyzer()
    df = sa.analyze_dataframe(df, text_col="content")

    summary = build_summary_report(df)
    assert summary["n_articles"] == len(df)
    assert "sentiment_by_category" in summary


def test_conversational_interface_returns_valid_response():
    df = load_news_data()
    sa = SentimentAnalyzer()
    df = sa.analyze_dataframe(df, text_col="content")

    qp = QueryProcessor(df, text_col="content")
    rg = ResponseGenerator()
    result = qp.process("Show me positive sport news")
    formatted = rg.format_response(result)
    assert "message" in formatted
    assert isinstance(formatted["n_results"], int)


def test_intent_classifier_detects_composite_intent():
    ic = IntentClassifier()
    parsed = ic.classify("Show me positive tech news from this week")
    assert parsed["intent"] == "filter_by_sentiment_category"
    assert parsed["entities"]["category"] == "tech"
    assert parsed["entities"]["sentiment"] == "positive"


def test_intent_classifier_detects_reset():
    ic = IntentClassifier()
    for phrase in ["clear filters", "start over", "reset", "new search"]:
        assert ic.classify(phrase)["intent"] == "reset"


def test_query_processor_explicit_context_is_isolated_per_caller():
    """
    Two independent callers passing their own prior_filters must never see
    or affect each other's results - this is what makes the Django app's
    per-session filtering safe for multiple concurrent users.
    """
    df = load_news_data()
    sa = SentimentAnalyzer()
    df = sa.analyze_dataframe(df, text_col="content")
    qp = QueryProcessor(df, text_col="content")

    result_a = qp.process("show me positive sport articles", prior_filters={})
    result_b = qp.process("show me tech news", prior_filters={})

    assert result_a["filters_applied"] == {"category": "sport", "sentiment": "positive"}
    assert result_b["filters_applied"] == {"category": "tech"}

    # A follow-up for caller A must only merge with A's own prior filters
    follow_up_a = qp.process("now show negative", prior_filters=result_a["filters_applied"])
    assert follow_up_a["filters_applied"] == {"category": "sport", "sentiment": "negative"}


def test_query_processor_reset_clears_filters():
    df = load_news_data()
    qp = QueryProcessor(df, text_col="content")
    prior = {"category": "tech", "sentiment": "positive"}
    result = qp.process("clear filters", prior_filters=prior)
    assert result["filters_applied"] == {}
    assert result["n_results"] == len(df)


def test_multilingual_pipeline_runs_end_to_end():
    ml_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "multilingual_sample.csv")
    df = pd.read_csv(ml_path)
    cla = CrossLingualAnalyzer()
    analyzed = cla.analyze_corpus(df)
    assert "language_name" in analyzed.columns
    assert "sentiment_label" in analyzed.columns
    coverage = cla.coverage_by_language(analyzed)
    assert len(coverage) > 0


def test_summarizer_produces_shorter_output():
    summarizer = Summarizer()
    text = ("The company reported strong quarterly earnings. Revenue grew by twelve percent "
            "year over year. Executives credited the growth to expansion in overseas markets. "
            "The stock price rose following the announcement. Analysts remain cautiously optimistic "
            "about the company's outlook for the remainder of the year.")
    summary = summarizer.summarize(text, n_sentences=2)
    assert len(summary) < len(text)
    assert len(summary) > 0


def test_embedding_index_semantic_search():
    docs = ["The football team won the championship.", "Stocks rallied on strong earnings.",
            "A new smartphone was released today.", "The soccer match ended in a draw."]
    idx = EmbeddingIndex()
    idx.build(docs)
    results = idx.most_similar("football game results", top_k=2)
    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]


def test_insight_generator_produces_text():
    gen = InsightGenerator()
    article = {"category": "tech", "sentiment_label": "positive",
               "entities": [{"text": "Microsoft", "label": "ORG"}], "confidence": 0.92}
    enhanced = gen.enhance_article(article)
    assert "Microsoft" in enhanced["enhancement"]
    assert "tech" in enhanced["enhancement"]

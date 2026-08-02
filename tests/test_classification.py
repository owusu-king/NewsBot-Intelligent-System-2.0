import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_processing.data_loader import load_news_data
from src.data_processing.text_preprocessor import TextPreprocessor
from src.data_processing.feature_extractor import FeatureExtractor
from src.analysis.classifier import NewsClassifier
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.ner_extractor import NERExtractor


def _prepared_data():
    df = load_news_data()
    pre = TextPreprocessor()
    df = pre.preprocess_dataframe(df, text_col="content", title_col="title")
    return df


def test_classifier_trains_and_selects_best_model():
    df = _prepared_data()
    fe = FeatureExtractor(max_features=500)
    X = fe.fit_transform(df["content_processed"])
    clf = NewsClassifier()
    results = clf.train(X, df["category"])
    assert clf.best_model_name in results
    assert results[clf.best_model_name]["f1"] > 0.5


def test_classifier_predict_with_confidence():
    df = _prepared_data()
    fe = FeatureExtractor(max_features=500)
    X = fe.fit_transform(df["content_processed"])
    clf = NewsClassifier()
    clf.train(X, df["category"])
    preds = clf.predict_with_confidence(X[:3])
    assert len(preds) == 3
    for p in preds:
        assert 0.0 <= p["confidence"] <= 1.0
        assert p["label"] in clf.classes_


def test_sentiment_analyzer_labels_are_valid():
    sa = SentimentAnalyzer()
    result = sa.analyze("This is a wonderful and amazing achievement!")
    assert result["label"] == "positive"
    result_neg = sa.analyze("This is a terrible and horrific disaster.")
    assert result_neg["label"] == "negative"


def test_ner_extractor_finds_entities():
    ner = NERExtractor()
    entities = ner.extract_entities("Apple Inc. announced a new product in California.")
    labels = {e["label"] for e in entities}
    assert "ORG" in labels or "GPE" in labels

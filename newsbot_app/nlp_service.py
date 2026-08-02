"""
nlp_service.py
----------------
Thin service layer connecting Django views to the src/ NLP package.
Uses module-level lazy singletons so expensive resources (spaCy model,
trained classifier, TF-IDF vectorizer, embedding index) are loaded once per
process, not once per request.

Model persistence: trained artifacts (classifier, vectorizer, topic model,
processed dataframe) are saved to data/models/newsbot_model.joblib after
training. On startup, a saved model is loaded instead of retraining from
scratch every time - training only happens when no saved model exists yet,
or when explicitly triggered via `python manage.py retrain_model`.
"""
import sys
import os
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_processing.data_loader import load_news_data, dataset_source
from src.data_processing.text_preprocessor import TextPreprocessor
from src.data_processing.feature_extractor import FeatureExtractor
from src.analysis.classifier import NewsClassifier
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.ner_extractor import NERExtractor
from src.analysis.topic_modeler import TopicModeler
from src.language_models.summarizer import Summarizer
from src.language_models.embeddings import EmbeddingIndex
from src.language_models.generator import InsightGenerator
from src.multilingual.language_detector import LanguageDetector
from src.multilingual.translator import Translator
from src.conversation.query_processor import QueryProcessor
from src.conversation.response_generator import ResponseGenerator
from src.utils.export import build_summary_report

MODEL_DIR = BASE_DIR / "data" / "models"
MODEL_PATH = MODEL_DIR / "newsbot_model.joblib"
MODEL_META_PATH = MODEL_DIR / "newsbot_model_meta.json"


class NewsBotService:
    """Singleton-style service. Loads a saved model if one exists; otherwise
    trains fresh (and saves the result so the next process start is fast)."""
    _instance = None

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor(max_features=3000)
        self.classifier = NewsClassifier()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.ner_extractor = NERExtractor()
        self.topic_modeler = TopicModeler(n_topics=6, method="lda")
        self.summarizer = Summarizer()
        self.generator = InsightGenerator()
        self.language_detector = LanguageDetector()
        self.translator = Translator()
        self.response_generator = ResponseGenerator()

        self.df = None
        self.query_processor = None
        self.model_trained_at = None

        if not self._load_artifacts():
            self._train()
            self._save_artifacts()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _train(self):
        df = load_news_data()
        df = self.preprocessor.preprocess_dataframe(df, text_col="content", title_col="title")
        df = self.sentiment_analyzer.analyze_dataframe(df, text_col="content")

        X = self.feature_extractor.fit_transform(df["content_processed"])
        self.classifier_results = self.classifier.train(X, df["category"])
        self.topic_modeler.fit_transform(df["content_processed"])
        df["topic_id"] = self.topic_modeler.dominant_topic_per_doc()

        self.df = df
        self.query_processor = QueryProcessor(df, text_col="content")
        self.dataset_source = dataset_source()
        self.model_trained_at = datetime.now(timezone.utc).isoformat()

    def _save_artifacts(self):
        """Persist everything needed to skip retraining on next startup."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "feature_extractor": self.feature_extractor,
            "classifier": self.classifier,
            "classifier_results": self.classifier_results,
            "topic_modeler": self.topic_modeler,
            "df": self.df,
            "dataset_source": self.dataset_source,
            "trained_at": self.model_trained_at,
        }, MODEL_PATH)
        MODEL_META_PATH.write_text(json.dumps({
            "trained_at": self.model_trained_at,
            "dataset_source": self.dataset_source,
            "n_articles": len(self.df),
            "best_classifier": self.classifier.best_model_name,
        }, indent=2))

    def _load_artifacts(self) -> bool:
        """Returns True if a saved model was found and loaded successfully."""
        if not MODEL_PATH.exists():
            return False
        try:
            saved = joblib.load(MODEL_PATH)
        except Exception:
            return False  # corrupt/incompatible save file -> fall back to training fresh

        self.feature_extractor = saved["feature_extractor"]
        self.classifier = saved["classifier"]
        self.classifier_results = saved["classifier_results"]
        self.topic_modeler = saved["topic_modeler"]
        self.df = saved["df"]
        self.dataset_source = saved["dataset_source"]
        self.model_trained_at = saved.get("trained_at")

        self.query_processor = QueryProcessor(self.df, text_col="content")
        return True

    def retrain(self, data_path: str = None):
        """Force a full retrain from current data and persist the result."""
        if data_path:
            df = load_news_data(path=data_path)
            df = self.preprocessor.preprocess_dataframe(df, text_col="content", title_col="title")
            df = self.sentiment_analyzer.analyze_dataframe(df, text_col="content")
            X = self.feature_extractor.fit_transform(df["content_processed"])
            self.classifier_results = self.classifier.train(X, df["category"])
            self.topic_modeler.fit_transform(df["content_processed"])
            df["topic_id"] = self.topic_modeler.dominant_topic_per_doc()
            self.df = df
            self.query_processor = QueryProcessor(df, text_col="content")
            self.dataset_source = dataset_source(path=data_path)
            self.model_trained_at = datetime.now(timezone.utc).isoformat()
        else:
            self._train()
        self._save_artifacts()

    def analyze_article(self, title: str, content: str) -> dict:
        clean = self.preprocessor.preprocess_text(content)
        X = self.feature_extractor.transform([clean])
        pred = self.classifier.predict_with_confidence(X)[0]

        sentiment = self.sentiment_analyzer.analyze(content)
        entities = self.ner_extractor.extract_entities(content)
        summary = self.summarizer.summarize(content, n_sentences=2)
        lang = self.language_detector.detect(content)

        translation = None
        if lang["language_code"] not in ("en", "unknown"):
            translation_result = self.translator.translate(content, source=lang["language_code"], target="en")
            translation = translation_result["translated_text"]

        article = {
            "title": title, "content": content,
            "category": pred["label"], "confidence": pred["confidence"],
            "sentiment_label": sentiment["label"], "sentiment_compound": sentiment["compound"],
            "entities": entities, "summary": summary,
            "language_code": lang["language_code"], "language_name": lang["language_name"],
            "translation": translation,
        }
        enhanced = self.generator.enhance_article(article)
        return enhanced

    def process_query(self, query_text: str, prior_filters: dict = None) -> dict:
        result = self.query_processor.process(query_text, prior_filters=prior_filters)
        response = self.response_generator.format_response(result)
        response["filters_applied"] = result["filters_applied"]
        return response

    def dashboard_summary(self) -> dict:
        return build_summary_report(self.df, self.classifier_results)

    def topics_overview(self):
        return self.topic_modeler.visualize_topics(n_words=8).to_dict("records")

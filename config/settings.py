"""
settings.py
------------
Central configuration for NewsBot 2.0. Reads secrets from environment
variables (see api_keys_template.txt) rather than hardcoding them.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Data paths ---
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "BBC News Train.csv"
SAMPLE_DATA_PATH = BASE_DIR / "data" / "sample" / "sample_news.csv"
MULTILINGUAL_SAMPLE_PATH = BASE_DIR / "data" / "sample" / "multilingual_sample.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "data" / "models"
RESULTS_DIR = BASE_DIR / "data" / "results"

# --- NLP settings ---
SPACY_MODEL = "en_core_web_sm"
TFIDF_MAX_FEATURES = 5000
N_TOPICS_DEFAULT = 8
TOPIC_MODEL_METHOD = "lda"  # or "nmf"
RANDOM_STATE = 42

# --- Multilingual settings ---
DEFAULT_TARGET_LANGUAGE = "en"
TRANSLATION_BACKEND = "google_translate"  # requires internet; falls back to offline demo glossary

# --- Optional API keys (loaded from environment, never hardcoded) ---
GOOGLE_TRANSLATE_API_KEY = os.environ.get("GOOGLE_TRANSLATE_API_KEY", "")
HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN", "")

# --- Django-specific (imported by newsbot_web/settings.py) ---
DJANGO_SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-in-production")
DJANGO_DEBUG = os.environ.get("DJANGO_DEBUG", "True") == "True"

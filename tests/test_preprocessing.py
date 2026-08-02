import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_processing.text_preprocessor import TextPreprocessor
from src.data_processing.data_validator import DataValidator
from src.data_processing.data_loader import load_news_data
import pandas as pd


def test_clean_text_removes_html_and_urls():
    pre = TextPreprocessor()
    dirty = "<p>Visit http://example.com for MORE info!! 123</p>"
    cleaned = pre.clean_text(dirty)
    assert "http" not in cleaned
    assert "<p>" not in cleaned
    assert cleaned == cleaned.lower()


def test_clean_text_handles_missing_values():
    pre = TextPreprocessor()
    assert pre.clean_text(None) == ""
    assert pre.clean_text(float("nan")) == ""


def test_preprocess_text_removes_stopwords():
    pre = TextPreprocessor(remove_stopwords=True)
    result = pre.preprocess_text("The cat is sitting on the mat")
    assert "the" not in result.split()
    assert "is" not in result.split()


def test_preprocess_dataframe_adds_expected_columns():
    df = pd.DataFrame({"content": ["Hello world"], "title": ["Hello"]})
    pre = TextPreprocessor()
    out = pre.preprocess_dataframe(df, text_col="content", title_col="title")
    assert "content_clean" in out.columns
    assert "content_processed" in out.columns
    assert "full_text" in out.columns


def test_data_validator_flags_missing_category():
    df = pd.DataFrame({"content": ["some text"], "category": [None]})
    validator = DataValidator()
    report = validator.validate(df)
    assert report["passed"] is False


def test_data_validator_passes_clean_data():
    df = pd.DataFrame({"content": ["some text here"], "category": ["tech"]})
    validator = DataValidator()
    report = validator.validate(df)
    assert report["passed"] is True


def test_load_news_data_standardizes_columns():
    df = load_news_data()
    assert set(["article_id", "title", "content", "category"]).issubset(df.columns)
    assert len(df) > 0

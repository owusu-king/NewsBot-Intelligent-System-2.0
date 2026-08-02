"""
data_loader.py
---------------
Loads the news dataset for NewsBot 2.0.

Priority order:
 1. data/raw/BBC News Train.csv (real Kaggle 'learn-ai-bbc' dataset, columns:
    ArticleId, Text, Category) if the student has downloaded it.
 2. data/sample/sample_news.csv (bundled offline sample so notebooks/tests
    always run end-to-end without a Kaggle account).

Standardizes columns to: article_id, content, category, title
"""
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "BBC News Train.csv")
SAMPLE_PATH = os.path.join(BASE_DIR, "data", "sample", "sample_news.csv")


def load_news_data(path: str = None) -> pd.DataFrame:
    """Load and standardize the news dataset."""
    if path is None:
        if os.path.exists(RAW_PATH):
            path = RAW_PATH
        elif os.path.exists(SAMPLE_PATH):
            path = SAMPLE_PATH
        else:
            raise FileNotFoundError(
                "No dataset found. Place the Kaggle BBC News Train.csv in data/raw/ "
                "or generate the sample dataset via data/sample/generate_sample_data.py"
            )

    df = pd.read_csv(path)

    # Standardize column names across BBC Kaggle format and sample format
    rename_map = {}
    if "ArticleId" in df.columns:
        rename_map["ArticleId"] = "article_id"
    if "Text" in df.columns:
        rename_map["Text"] = "content"
    if "Category" in df.columns:
        rename_map["Category"] = "category"
    df = df.rename(columns=rename_map)

    if "title" not in df.columns:
        # BBC dataset has no title column - synthesize one from first sentence
        df["title"] = df["content"].astype(str).apply(
            lambda t: (t.split(".")[0][:80] + "...") if len(t) > 80 else t.split(".")[0]
        )

    if "article_id" not in df.columns:
        df["article_id"] = range(1, len(df) + 1)

    df["category"] = df["category"].astype(str).str.lower().str.strip()
    return df[["article_id", "title", "content", "category"]]


def dataset_source(path: str = None) -> str:
    """Report which dataset is currently active (for transparency in notebooks/UI)."""
    if path:
        return path
    if os.path.exists(RAW_PATH):
        return f"Real BBC Kaggle dataset: {RAW_PATH}"
    return f"Bundled sample dataset (demo only): {SAMPLE_PATH}"

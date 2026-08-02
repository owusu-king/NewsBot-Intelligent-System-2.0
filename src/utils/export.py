"""
export.py
----------
Report generation / export utilities: dump analysis results to CSV/JSON so
the Django app and notebooks can produce downloadable reports on demand.
"""
import json
import pandas as pd


def export_to_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    return path


def export_to_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def build_summary_report(df: pd.DataFrame, classifier_results: dict = None) -> dict:
    """Aggregate the numbers needed for the executive summary / dashboard."""
    report = {
        "n_articles": len(df),
        "category_counts": df["category"].value_counts().to_dict() if "category" in df else {},
    }
    if "sentiment_compound" in df.columns:
        report["sentiment_by_category"] = df.groupby("category")["sentiment_compound"].mean().to_dict()
        report["overall_sentiment_avg"] = round(df["sentiment_compound"].mean(), 4)
    if classifier_results:
        best = max(classifier_results, key=lambda n: classifier_results[n]["f1"])
        report["best_classifier"] = best
        report["classifier_f1"] = round(classifier_results[best]["f1"], 4)
        report["classifier_accuracy"] = round(classifier_results[best]["accuracy"], 4)
    return report

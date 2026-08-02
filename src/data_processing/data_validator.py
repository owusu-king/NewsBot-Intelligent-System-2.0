"""
data_validator.py
------------------
Basic data quality checks run before analysis: missing values, empty text,
duplicate articles, and category balance.
"""
import pandas as pd


class DataValidator:
    """Runs a set of sanity checks on a news dataframe and reports issues."""

    REQUIRED_COLUMNS = ["content", "category"]

    def validate(self, df: pd.DataFrame) -> dict:
        report = {"passed": True, "issues": [], "warnings": [], "stats": {}}

        # Required columns
        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            report["passed"] = False
            report["issues"].append(f"Missing required columns: {missing_cols}")
            return report

        # Missing values
        n_missing_content = df["content"].isna().sum() + (df["content"].astype(str).str.strip() == "").sum()
        n_missing_category = df["category"].isna().sum()
        if n_missing_content > 0:
            report["warnings"].append(f"{n_missing_content} rows with empty/missing content")
        if n_missing_category > 0:
            report["issues"].append(f"{n_missing_category} rows with missing category")
            report["passed"] = False

        # Duplicates
        n_dupes = df.duplicated(subset=["content"]).sum()
        if n_dupes > 0:
            report["warnings"].append(f"{n_dupes} duplicate articles found")

        # Very short articles (likely noise)
        short_articles = (df["content"].astype(str).str.split().str.len() < 5).sum()
        if short_articles > 0:
            report["warnings"].append(f"{short_articles} articles with fewer than 5 words")

        # Category balance
        counts = df["category"].value_counts()
        report["stats"]["category_counts"] = counts.to_dict()
        if counts.max() / max(counts.min(), 1) > 5:
            report["warnings"].append("Category distribution is highly imbalanced (max/min ratio > 5)")

        report["stats"]["n_rows"] = len(df)
        report["stats"]["n_categories"] = df["category"].nunique()
        return report

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows that fail hard validation (missing category/content, dupes)."""
        df = df.copy()
        df = df.dropna(subset=["category"])
        df = df[df["content"].astype(str).str.strip() != ""]
        df = df.drop_duplicates(subset=["content"])
        return df.reset_index(drop=True)

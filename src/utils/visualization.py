"""
visualization.py
------------------
Reusable plotting functions for notebooks (matplotlib/seaborn) so every
notebook doesn't re-implement the same chart code.
"""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_category_distribution(df, category_col="category", ax=None):
    ax = ax or plt.gca()
    counts = df[category_col].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title("Article Count by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_sentiment_by_category(df, category_col="category", sentiment_col="sentiment_compound", ax=None):
    ax = ax or plt.gca()
    sns.boxplot(data=df, x=category_col, y=sentiment_col, ax=ax)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Sentiment Distribution by Category")
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_confusion_matrix(cm, labels, ax=None):
    ax = ax or plt.gca()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return ax


def plot_topic_words(topic_words_dict, n_cols=3):
    n_topics = len(topic_words_dict)
    n_rows = (n_topics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    axes = axes.flatten() if n_topics > 1 else [axes]
    for i, (topic_id, words) in enumerate(topic_words_dict.items()):
        axes[i].barh(words[::-1], range(1, len(words) + 1))
        axes[i].set_title(f"Topic {topic_id}")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    return fig


def plot_language_distribution(df, lang_col="language_name", ax=None):
    ax = ax or plt.gca()
    counts = df[lang_col].value_counts()
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
    ax.set_title("Article Distribution by Language")
    return ax

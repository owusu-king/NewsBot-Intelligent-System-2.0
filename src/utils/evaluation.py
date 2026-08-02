"""
evaluation.py
--------------
Shared model evaluation utilities used across analysis modules and tests.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_summary(y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": round(acc, 4), "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def topic_coherence_proxy(topic_word_lists, corpus_tokens):
    """
    A lightweight topic coherence proxy (co-occurrence based) that doesn't
    require gensim's full coherence pipeline - useful for quick topic-model
    quality comparisons across n_topics/method choices in the notebooks.
    """
    doc_sets = [set(tokens) for tokens in corpus_tokens]
    scores = []
    for words in topic_word_lists:
        pair_scores = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                co_occur = sum(1 for d in doc_sets if words[i] in d and words[j] in d)
                occur_j = sum(1 for d in doc_sets if words[j] in d)
                if occur_j > 0:
                    pair_scores.append(np.log((co_occur + 1) / occur_j))
        scores.append(np.mean(pair_scores) if pair_scores else 0.0)
    return scores


def confidence_calibration_bins(confidences, correct, n_bins=5):
    """Bucket predictions by confidence to check if the model is well-calibrated."""
    bins = np.linspace(0, 1, n_bins + 1)
    results = []
    confidences = np.array(confidences)
    correct = np.array(correct)
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() > 0:
            results.append({
                "bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                "n": int(mask.sum()),
                "avg_confidence": round(float(confidences[mask].mean()), 3),
                "accuracy": round(float(correct[mask].mean()), 3),
            })
    return results

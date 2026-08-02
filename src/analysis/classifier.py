"""
classifier.py
--------------
Enhanced news classification system (builds on midterm Module 7).
Compares Naive Bayes / Logistic Regression / SVM, selects the best model,
and exposes confidence-scored predictions.
"""
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


class NewsClassifier:
    """Trains and compares multiple classifiers, keeps the best one."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.candidates = {
            "Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
            "SVM": SVC(random_state=random_state, probability=True),
        }
        self.trained_models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        self.classes_ = None

    def train(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        for name, clf in self.candidates.items():
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")
            self.trained_models[name] = clf
            self.results[name] = {"accuracy": acc, "f1": f1, "predictions": preds}

        self.best_model_name = max(self.results, key=lambda n: self.results[n]["f1"])
        self.best_model = self.trained_models[self.best_model_name]
        self.classes_ = self.best_model.classes_
        self._X_test, self._y_test = X_test, y_test
        return self.results

    def evaluation_report(self):
        preds = self.results[self.best_model_name]["predictions"]
        return {
            "best_model": self.best_model_name,
            "classification_report": classification_report(self._y_test, preds, zero_division=0, output_dict=True),
            "confusion_matrix": confusion_matrix(self._y_test, preds, labels=self.classes_).tolist(),
            "labels": list(self.classes_),
        }

    def predict(self, X):
        return self.best_model.predict(X)

    def predict_with_confidence(self, X):
        """Return (label, confidence, per_class_probabilities) for each row."""
        preds = self.best_model.predict(X)
        if hasattr(self.best_model, "predict_proba"):
            probs = self.best_model.predict_proba(X)
        else:
            # fall back to decision_function normalized
            scores = self.best_model.decision_function(X)
            probs = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
        results = []
        for i, label in enumerate(preds):
            class_probs = dict(zip(self.classes_, probs[i]))
            confidence = float(max(probs[i]))
            results.append({"label": label, "confidence": confidence, "class_probabilities": class_probs})
        return results

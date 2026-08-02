"""
topic_modeler.py
------------------
LDA and NMF topic modeling for content discovery, following the exact
interface sketched in the assignment brief (fit_transform / get_topic_words /
visualize_topics), plus topic evolution over time and content clustering.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TopicModeler:
    def __init__(self, n_topics=10, method="lda", max_features=2000):
        self.n_topics = n_topics
        self.method = method
        self.max_features = max_features
        self.model = None
        self.vectorizer = None
        self.doc_topic_matrix = None

    def fit_transform(self, documents):
        """Train topic model and transform documents into topic distributions."""
        if self.method == "lda":
            self.vectorizer = CountVectorizer(max_features=self.max_features, stop_words="english", min_df=2)
            dtm = self.vectorizer.fit_transform(documents)
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics, random_state=42, learning_method="online"
            )
        elif self.method == "nmf":
            self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words="english", min_df=2)
            dtm = self.vectorizer.fit_transform(documents)
            self.model = NMF(n_components=self.n_topics, random_state=42, init="nndsvd", max_iter=500)
        else:
            raise ValueError("method must be 'lda' or 'nmf'")

        self.doc_topic_matrix = self.model.fit_transform(dtm)
        return self.doc_topic_matrix

    def get_topic_words(self, topic_id, n_words=10):
        """Get top words for a specific topic."""
        if self.model is None:
            raise RuntimeError("Call fit_transform() first.")
        feature_names = self.vectorizer.get_feature_names_out()
        topic = self.model.components_[topic_id]
        top_indices = topic.argsort()[::-1][:n_words]
        return [feature_names[i] for i in top_indices]

    def get_all_topics(self, n_words=10):
        return {i: self.get_topic_words(i, n_words) for i in range(self.n_topics)}

    def dominant_topic_per_doc(self):
        return self.doc_topic_matrix.argmax(axis=1)

    def topic_evolution(self, dates, freq="M"):
        """Track how topic prevalence changes over time. `dates` aligned with docs."""
        if self.doc_topic_matrix is None:
            raise RuntimeError("Call fit_transform() first.")
        df = pd.DataFrame(self.doc_topic_matrix, columns=[f"topic_{i}" for i in range(self.n_topics)])
        df["date"] = pd.to_datetime(dates)
        return df.set_index("date").resample(freq).mean()

    def cluster_by_topic(self, threshold=0.3):
        """Group documents into clusters based on their dominant topic distribution."""
        dominant = self.dominant_topic_per_doc()
        clusters = {}
        for topic_id in range(self.n_topics):
            clusters[topic_id] = np.where(dominant == topic_id)[0].tolist()
        return clusters

    def visualize_topics(self, n_words=8):
        """
        Create a simple textual topic summary (Colab-safe fallback; the
        Django dashboard and notebooks render this as an interactive chart
        via pyLDAvis / plotly where available).
        """
        summary = []
        for topic_id, words in self.get_all_topics(n_words).items():
            summary.append({"topic": topic_id, "top_words": ", ".join(words)})
        return pd.DataFrame(summary)

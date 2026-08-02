"""
embeddings.py
--------------
Semantic embeddings for similarity search and query expansion.

Default backend is TF-IDF + cosine similarity, which is lightweight, fully
offline, and works well for a corpus-sized (hundreds-thousands of articles)
news search index. An optional sentence-transformers backend is provided for
production use when internet access to the HuggingFace hub is available.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingIndex:
    def __init__(self, backend="tfidf"):
        self.backend = backend
        self.vectorizer = None
        self.doc_vectors = None
        self.documents = None
        self._st_model = None

    def build(self, documents):
        self.documents = list(documents)
        if self.backend == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            self.doc_vectors = self.vectorizer.fit_transform(self.documents)
        elif self.backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer  # optional dependency
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.doc_vectors = self._st_model.encode(self.documents)
        else:
            raise ValueError("backend must be 'tfidf' or 'sentence_transformers'")
        return self.doc_vectors

    def _embed_query(self, query):
        if self.backend == "tfidf":
            return self.vectorizer.transform([query])
        return self._st_model.encode([query])

    def most_similar(self, query, top_k=5):
        """Semantic search: return top_k most similar documents to the query."""
        q_vec = self._embed_query(query)
        sims = cosine_similarity(q_vec, self.doc_vectors).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [{"index": int(i), "score": float(sims[i]), "document": self.documents[i]} for i in top_idx]

    def similar_articles(self, doc_index, top_k=5):
        """Find articles most similar to a given article (content-based recommendation)."""
        sims = cosine_similarity(self.doc_vectors[doc_index], self.doc_vectors).flatten()
        top_idx = [i for i in np.argsort(sims)[::-1] if i != doc_index][:top_k]
        return [{"index": int(i), "score": float(sims[i])} for i in top_idx]

    def expand_query(self, query, n_terms=5):
        """
        Query expansion: for the TF-IDF backend, find terms in the vocabulary
        most associated with the query's own top terms (co-occurring highly-
        weighted terms across the corpus), to improve recall for short queries.
        """
        if self.backend != "tfidf":
            return [query]
        q_vec = self.vectorizer.transform([query])
        feature_names = self.vectorizer.get_feature_names_out()
        top_query_terms = [feature_names[i] for i in q_vec.toarray()[0].argsort()[::-1][:3] if q_vec.toarray()[0][i] > 0]

        # Terms that co-occur with the query terms in the most similar documents
        neighbor_docs = self.most_similar(query, top_k=10)
        neighbor_indices = [d["index"] for d in neighbor_docs]
        sub_matrix = self.doc_vectors[neighbor_indices]
        mean_scores = np.asarray(sub_matrix.mean(axis=0)).flatten()
        top_expansion_idx = mean_scores.argsort()[::-1][:n_terms + len(top_query_terms)]
        expansion_terms = [feature_names[i] for i in top_expansion_idx if feature_names[i] not in top_query_terms]
        return list(dict.fromkeys(top_query_terms + expansion_terms[:n_terms]))

"""
ner_extractor.py
------------------
Named entity recognition (builds on midterm Module 8), extended with basic
entity relationship mapping (co-occurrence within the same article/sentence)
for the final project's "Entity Relationship Mapping" requirement.
"""
from collections import Counter, defaultdict
from itertools import combinations
import spacy
import pandas as pd


class NERExtractor:
    def __init__(self, model="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{model}' not found. Run: python -m spacy download {model}"
            )

    def extract_entities(self, text: str) -> list:
        if not text or pd.isna(text):
            return []
        doc = self.nlp(str(text)[:10000])
        return [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                for ent in doc.ents]

    def extract_dataframe(self, df: pd.DataFrame, text_col="content", id_col="article_id") -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            for ent in self.extract_entities(row[text_col]):
                ent[id_col] = row[id_col]
                ent["category"] = row.get("category", None)
                rows.append(ent)
        return pd.DataFrame(rows)

    def entity_relationship_map(self, df: pd.DataFrame, text_col="content", min_count=2):
        """
        Build a co-occurrence graph: entities that appear in the same article
        are considered "related". Returns edges with weights (# of shared articles).
        """
        edge_weights = Counter()
        node_labels = {}
        for _, row in df.iterrows():
            ents = self.extract_entities(row[text_col])
            unique_ents = list({(e["text"], e["label"]) for e in ents if e["label"] in
                                 ("PERSON", "ORG", "GPE", "NORP", "EVENT")})
            for (t1, l1), (t2, l2) in combinations(unique_ents, 2):
                edge = tuple(sorted([t1, t2]))
                edge_weights[edge] += 1
                node_labels[t1] = l1
                node_labels[t2] = l2

        edges = [{"source": a, "target": b, "weight": w, "source_label": node_labels.get(a),
                  "target_label": node_labels.get(b)}
                 for (a, b), w in edge_weights.items() if w >= min_count]
        return sorted(edges, key=lambda e: -e["weight"])

    def entity_sentiment_association(self, entities_df: pd.DataFrame, sentiment_by_article: dict, id_col="article_id"):
        """Average sentiment of articles in which each entity appears."""
        assoc = defaultdict(list)
        for _, row in entities_df.iterrows():
            score = sentiment_by_article.get(row[id_col])
            if score is not None:
                assoc[(row["text"], row["label"])].append(score)
        return {k: sum(v) / len(v) for k, v in assoc.items() if len(v) > 0}

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_processing.data_loader import load_news_data
from src.data_processing.text_preprocessor import TextPreprocessor
from src.analysis.topic_modeler import TopicModeler


def _processed_texts():
    df = load_news_data()
    pre = TextPreprocessor()
    df = pre.preprocess_dataframe(df, text_col="content", title_col="title")
    return df["content_processed"]


def test_lda_topic_modeler_fits_and_returns_words():
    texts = _processed_texts()
    tm = TopicModeler(n_topics=4, method="lda")
    doc_topics = tm.fit_transform(texts)
    assert doc_topics.shape[0] == len(texts)
    assert doc_topics.shape[1] == 4
    words = tm.get_topic_words(0, n_words=5)
    assert len(words) == 5


def test_nmf_topic_modeler_fits_and_returns_words():
    texts = _processed_texts()
    tm = TopicModeler(n_topics=4, method="nmf")
    doc_topics = tm.fit_transform(texts)
    assert doc_topics.shape[1] == 4
    all_topics = tm.get_all_topics(n_words=5)
    assert len(all_topics) == 4


def test_cluster_by_topic_partitions_all_documents():
    texts = _processed_texts()
    tm = TopicModeler(n_topics=3, method="lda")
    tm.fit_transform(texts)
    clusters = tm.cluster_by_topic()
    total_docs = sum(len(v) for v in clusters.values())
    assert total_docs == len(texts)

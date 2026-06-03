"""AutoDL artifact generation behavior contracts."""

from __future__ import annotations

import numpy as np


def test_medcpt_corpus_embedding_formats_title_content_pairs() -> None:
    from app.rag.experiments import run_medcpt_embedding_autodl as module

    formatted = module._format_medcpt_article_inputs(
        [{"title": "Title", "content": "Content"}]
    )

    assert formatted == [["Title", "Content"]]


def test_local_bge_embedding_batches_long_corpus_texts() -> None:
    from app.rag.experiments import run_local_bge_embedding_autodl as module

    class FakeEmbedding:
        def __init__(self):
            self.batch_sizes = []

        def get_text_embedding_batch(self, texts, show_progress):
            self.batch_sizes.append(len(texts))
            return [[1.0, 0.0] for _ in texts]

    fake_embedding = FakeEmbedding()
    embeddings = module.embed_texts(
        fake_embedding,
        ["a", "b", "c", "d", "e"],
        batch_size=2,
        progress_label="test",
    )

    assert fake_embedding.batch_sizes == [2, 2, 1]
    assert embeddings.shape == (5, 2)
    assert np.allclose(embeddings[:, 0], 1.0)


def test_local_rerank_cache_rows_are_serializable_artifacts() -> None:
    from app.rag.evaluation.formal_local_rerank_cache import LOCAL_RERANKER_BACKEND
    from app.rag.experiments import run_local_rerank_cache_autodl as module

    class FakeReranker:
        def postprocess_nodes(self, nodes, query_str):
            assert query_str == "query"
            return nodes[:1]

    rows = module.rerank_cache_rows(
        "cache-a",
        [
            {
                "question_id": "dev-1",
                "query_text": "query",
                "candidates": [{"text": "context", "score": 0.5}],
            }
        ],
        FakeReranker(),
    )

    assert rows[0]["reranker_backend"] == LOCAL_RERANKER_BACKEND
    assert rows[0]["reranked_candidates"][0]["text"] == "context"

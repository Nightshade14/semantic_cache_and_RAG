from .vectorstore import Vectorstore
from .oollama import Model
from .semantic_cache import SemanticCache


import numpy as np

model = Model()

test_emb = model.get_embeddings("test")
vectorstore = Vectorstore()

semantic_cache = SemanticCache(Model())
semantic_cache.vector_store.add(test_emb, "test")


def test_vectorstore_add():
    vectorstore.add(test_emb, "test")
    assert_emb, assert_doc = vectorstore.vector_store[0]
    assert assert_doc == "test"
    assert np.array_equal(assert_emb, test_emb)


def test_vectorstore_search():
    search_res = vectorstore.search(test_emb)
    assert search_res[0].doc == "test"
    assert isinstance(search_res[0].score, np.float64)


def test_semantic_cache_query():
    res = semantic_cache.query("test")
    assert res.answer == "test"
    assert res.hit == True

    res = semantic_cache.query("Apple")
    assert res.answer != "test"
    assert res.hit == False

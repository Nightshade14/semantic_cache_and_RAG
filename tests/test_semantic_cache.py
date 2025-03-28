from src.oollama import Model
from src.semantic_cache import SemanticCache

model = Model()
test_emb = model.get_embeddings("test")

semantic_cache = SemanticCache(model=model)
semantic_cache.vector_store.add(test_emb, "test")


def test_semantic_cache_query():
    res = semantic_cache.query("test")
    assert res.answer == "test"
    assert res.hit == True

    res = semantic_cache.query("Apple")
    assert res.answer != "test"
    assert res.hit == False

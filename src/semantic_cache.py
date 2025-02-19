from .oollama import Model
from .vectorstore import Vectorstore
from dataclasses import dataclass
import numpy as np


@dataclass
class SemanticCacheResult:
    answer: str
    hit: bool


class SemanticCache:
    def __init__(self, model: Model, n=5):
        self.model = model
        self.vector_store = Vectorstore()
        self.n = n

    def query(self, query: str, threshold=20) -> SemanticCacheResult:
        """
        Answer the user query `self.model.llm(query)`
        First check to see if a similar query has been asked and use the cached query if it exists.

          Examples:
        query("what is the capital of france") -> SemanticCacheResult("The capital is Paris", False)
        query("what is france's capital") -> SemanticCacheResult("The capital is Paris", True)

        """
        model = Model()
        query_embedding = np.array(model.get_embeddings(query))

        result = self.vector_store.search(embedding=query_embedding, k=1)
        cached_answer = result[0].doc
        distance = result[0].score
        answer = None
        hit = None
        # print(f"Semantic cache score for {query}: {distance}\n\n")

        if len(result) and distance < threshold:
            answer = cached_answer
            hit = True
        else:
            answer = np.array(model.answer_query(query=query))
            hit = False

        return SemanticCacheResult(answer=answer, hit=hit)

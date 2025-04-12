import heapq
from typing import List
from dataclasses import dataclass
import numpy as np


@dataclass
class VectorSearchResult:
    doc: str
    score: float


class Vectorstore:
    def __init__(self):
        self.vector_store = []

    def add(self, embedding: List[float], doc: str):
        """
        Adds a document to the vector store.
        """

        embedding = np.array(embedding)
        self.vector_store.append((embedding, doc))

    def search(self, embedding: List[float], k: int = 5) -> List[VectorSearchResult]:
        """
        Search through vector store and return top k elements.
        Uses euclidian distance sqrt(sum((doc_embedding[i]-query_embedding[i])^2))

        :return List of k vectors search results containing retrieved doc and score
        """

        embedding = np.array(embedding)
        vector_db = []

        for vector, doc in self.vector_store:
            distance = np.linalg.norm(vector - embedding)
            heapq.heappush(vector_db, (distance, doc))

        end = k
        if len(vector_db) < k:
            end = len(vector_db)

        results = heapq.nsmallest(end, vector_db)

        return [
            VectorSearchResult(doc=result[1], score=result[0]) for result in results
        ]

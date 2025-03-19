# Vector Store from Scratch, Semantic Caching System, FAISS and Spotify's Voyager RAG

An efficient implementation of  the following:
- Vector Store from scratch
- Semantic Caching for minimizing LLM API calls
- RAG with FAISS as in-memory vector store


## 🚀 Features

### 1. RAG with FAISS Vector Store
- Fast similarity search using FAISS (Facebook AI Similarity Search)
- Document chunking and vectorization
- Efficient storage and retrieval of embeddings
- Asynchronous document processing

### 2. Semantic Cache
- Smart caching system for similar queries
- Cache hit optimization for frequently asked questions

### 3. Implementing Vector Store from Scratch
- Priority queue implementation for top-k retrieval


## 🛠️ Technologies Used
- Fine-tune FAISS for vector similarity search (try with spotify's voyager)
- LangChain for document processing
- Python's heapq for priority queue implementation
- NumPy for vector operations

## 📦 Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/yourusername/semantic_cache_and_RAG.git
cd semantic_cache_and_RAG
uv sync
```

## 📝 Testing

```bash
uv run pytest .
```

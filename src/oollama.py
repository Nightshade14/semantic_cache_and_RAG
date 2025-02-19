import requests
# import asyncio


class Model:
    def __init__(self, embedding_model_url="http://localhost:11434/api/embeddings", llm_url="http://localhost:11434/api/generate"):
        self.embedding_model_url = embedding_model_url
        self.llm_url = llm_url

    def get_embeddings(self, query):
        response = requests.post(
            self.embedding_model_url,
            json={"model": "nomic-embed-text", "prompt": query},
        )

        if response.status_code == 200:
            return response.json().get("embedding")
        else:
            response.raise_for_status()

    def answer_query(self, query):
        payload = {"model": "phi4", "prompt": query, "stream": False}
        response = requests.post(self.llm_url, json=payload)
        # response = await asyncio.to_thread(requests.post, self.llm_url, json=payload)
        if response.status_code == 200:
            return response.json().get("response")
        else:
            response.raise_for_status()


# async def main():
#     embeddings = model.get_embeddings("Sample text")
#     ans = await asyncio.create_task(model.answer_query("What is the capital of France?"))
#     print(f"ans: {ans}")


# Example usage:
# model = Model(
#     embedding_model_url="http://localhost:11434/api/embeddings",
#     llm_url="http://localhost:11434/api/generate",
# )


# embeddings = model.get_embeddings("Sample text")
# ans = model.answer_query("What is the capital of France?")

# print(f"ans: {ans}")

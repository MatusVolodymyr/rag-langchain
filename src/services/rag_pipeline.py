import openai
from src.services.vector_store import VectorStore


class RAGPipeline:
    def __init__(self, vector_store: VectorStore, openai_api_key):
        """
        Initializes the RAG pipeline with vector storage and OpenAI API key.
        """
        self.vector_store = vector_store
        openai.api_key = openai_api_key

    def generate_response(self, query, top_k=3):
        """
        Retrieves relevant documents and generates a response using LLM.
        """
        retrieved_docs = self.vector_store.retrieve_documents(query, top_k)
        context = (
            "\n".join(retrieved_docs)
            if retrieved_docs
            else "No relevant documents found."
        )

        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that answers questions based on provided context.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response["choices"][0]["message"]["content"]

import os
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader


class VectorStore:
    def __init__(
        self, index_name, api_key, environment, chunk_size=256, chunk_overlap=50
    ):
        """
        Initialize Pinecone connection and embedding model.
        """
        pc = Pinecone(api_key)
        if index_name not in pc.list_indexes():
            self.index = pc.Index(index_name)
        else:
            raise ValueError(
                f"Index '{index_name}' does not exist. Please create it first."
            )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Text chunking settings
        self.text_splitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def add_documents(self, documents):
        """
        Embed and store documents in Pinecone.

        :param documents: List of dictionaries {"id": str, "text": str}
        """
        vectors = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc["text"])  # Chunking
            for i, chunk in enumerate(chunks):
                vector = self.embedding_model.embed_query(chunk)
                vectors.append(
                    (f"{doc['id']}_chunk{i}", vector, {"text": chunk})
                )  # Store chunked text ("123_chunk0", [0.1, 0.2, 0.3], {"text": "This is a long document"})

        self.index.upsert(vectors)
        print(f"Stored {len(vectors)} chunks in Pinecone.")

    def retrieve_documents(self, query, top_k=3):
        """
        Retrieve the most relevant documents based on a query.

        :param query: User's search query
        :param top_k: Number of documents to retrieve
        :return: List of relevant document texts
        """
        query_embedding = self.embedding_model.embed_query(query)
        results = self.index.query(query_embedding, top_k=top_k, include_metadata=True)

        return [match["metadata"]["text"] for match in results["matches"]]

    def process_file(self, file_path):
        """
        Load and extract text from a .txt or .pdf file.
        """
        ext = file_path.split(".")[-1]
        if ext == "txt":
            loader = TextLoader(file_path)
        elif ext == "pdf":
            loader = PyMuPDFLoader(file_path)  # PDF extraction
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        docs = loader.load()
        return docs  # Returns list of LangChain Document objects

    def add_documents_from_file(self, file_path):
        """
        Extract text from file, chunk it, embed, and store in Pinecone.
        """
        docs = self.process_file(file_path)
        # texts = [doc.page_content for doc in docs]
        chunks = self.text_splitter.split_documents(docs)  # Chunk text

        vectors = []
        for i, chunk in enumerate(chunks):
            vector = self.embedding_model.embed_query(
                chunk.page_content
            )  # Extract text
            vectors.append(
                (
                    f"{os.path.basename(file_path)}_chunk{i}",
                    vector,
                    {"text": chunk.page_content, **chunk.metadata},
                )
            )

        self.index.upsert(vectors)
        print(f"Stored {len(vectors)} chunks from {file_path} in Pinecone.")

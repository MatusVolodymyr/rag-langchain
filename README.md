# RAG System API

This is a Retrieval-Augmented Generation (RAG) system API built with FastAPI. It retrieves relevant documents from a vector store (Pinecone) and generates responses using an LLM (OpenAI's GPT-3.5-turbo). This project is primarily for learning and portfolio purposes.

## Project Structure
```
.
├───Dockerfile
├───environment.yml
├───main.py
├───pytest.ini
├───READMR.md
│
├───examples
│       example.pdf
│       example.txt
│
├───src
│   │   dependencies.py
│   │
│   ├───api
│   │   └───endpoints
│   │       ├───documents.py
│   │       ├───query.py
│   │       └───response.py
│   │
│   ├───core
│   │   ├───config.py
│   │   ├───logger.py
│   │   └───__init__.py
│   │
│   ├───models
│   │   └───document.py
│   │
│   └───services
│       ├───rag_pipeline.py
│       └───vector_store.py
│
└───tests
    └───test_query.py
```

## Features

- **Document Upload:** Upload and process text and PDF files.
- **Query Endpoint:** Retrieve relevant documents based on user queries.
- **Response Generation:** Generate responses using an LLM based on the retrieved context.
- **Dependency Injection:** Utilizes FastAPI's dependency injection for cleaner code.
- **Asynchronous Processing:** Offloads heavy tasks using FastAPI's BackgroundTasks.
- **Testing:** Includes basic tests with FastAPI's TestClient and pytest.
- **Containerization:** Dockerfile provided for containerizing the application with a Conda environment.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Conda (or Mamba) for environment management
- Docker (optional, for containerization)

### Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create the Conda Environment**

   Create and activate the environment:

   ```bash
   conda env create -f environment.yml
   conda activate <your-environment-name>
   ```



### Running the Application

#### Locally with Uvicorn

Start the FastAPI server with:

```bash
uvicorn main:app --reload
```

The API will be accessible at [http://127.0.0.1:8000](http://127.0.0.1:8000).

#### With Docker

🚧🚧In construction🚧🚧

The API will be available at [http://localhost:8000](http://localhost:8000).

### Running Tests

Run the test suite with:

```bash
pytest
```

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please open an issue on GitHub.


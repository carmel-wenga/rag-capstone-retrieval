# Utrains RAG Lab

This lab is a hands-on project that demonstrates a beginner-friendly Retrieval-Augmented Generation (RAG) workflow for 
a Utrains support chatbot. Students can index a small Q&A dataset into Elasticsearch, use full-text or vector search, 
and ask an LLM to answer using only the retrieved context.

## Project goals

- Store support questions and answers in Elasticsearch.
- Generate embeddings for the question text with LangChain's OpenAI integration.
- Search using either full-text search or vector similarity.
- Pass the retrieved context to an LLM to produce a grounded answer.
- Run the whole system locally with Docker Compose.

## Project structure

```text
utrains-rag-lab/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .env.example
├── README.md
├── data/
│   └── source_dataset.json
├── offline/
│   └── index_data.py
├── app.py
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── llm_client.py
│   └── rag.py
```

## Step-by-step setup

Follow these steps from the project root.

### 1. Install the project dependencies

If you do not have `uv` installed yet, install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project dependencies:

```bash
cd utrains-rag-lab
uv sync
```

### 2. Configure the project secrets

Create a local secrets file at `.streamlit/secrets.toml` with your OpenAI API key and Elasticsearch connection details used by the LangChain OpenAI integration:

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
ELASTICSEARCH_HOST = "http://localhost:9200"
INDEX_NAME = "utrains-qa"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
VECTOR_DIMENSION = 1536
STREAMLIT_PORT = 8501
```

The app reads values with `st.secrets["variable_name"]` when it runs in Streamlit, and it falls back to this file 
when you run scripts outside the app.

### 3. Start Elasticsearch

You can run Elasticsearch with Docker Compose:

```bash
docker compose up -d vectordatabase
```

This starts a single-node Elasticsearch instance with security disabled for the lab.

If you already have Elasticsearch running locally, make sure it is available at `http://localhost:9200` and skip this step.

### 4. Index the dataset

The offline indexing script loads a dataset, embeds each question through `langchain-openai`, and stores the results in Elasticsearch.

Use the dataset file you want to index. For the sample source dataset in this repository:

```bash
docker compose exec streamlit uv run python -m offline.index_data data/source_dataset.json
```

This is the command used for the data indexing in this project.

You should see logs showing:

- creation of the `utrains-qa` index,
- embedding generation for each question,
- bulk indexing of the documents,
- a final total count of indexed records.

### 5. Run the Streamlit app

From the project root, start the web app:

```bash
uv run streamlit run app.py
```

The app will open in your browser. Use the sidebar to switch between:

- Vector Search (Embeddings)
- Full-Text Search (BM25)

Then ask questions such as:

- "What is the duration of the DevOps training program?"
- "Does Utrains provide internship placement support?"
- "How do I reset my learner account password?"

The app shows both the chatbot response and the retrieved context in an expandable section so you can inspect what the model saw.

### 6. Run the full Docker stack

To start both Elasticsearch and the Streamlit app in one command:

```bash
docker compose up --build
```

This is the easiest setup for a classroom or local demo environment, and the app waits for Elasticsearch to be ready before starting.

## 7. How the RAG flow works

1. The user sends a question to the app.
2. The app chooses either full-text or vector search.
3. Elasticsearch retrieves the most relevant Q&A chunks.
4. Those chunks are added to the prompt as context.
5. The LangChain chat model backed by OpenAI answers using only the context it has received.
6. The UI displays the response and the retrieved context for transparency.

## Troubleshooting

### OpenAI API key error

- Make sure `.streamlit/secrets.toml` contains a valid `openai_api_key` value.
- Ensure the key is not blank and is active for your OpenAI account.

### Elasticsearch connection error

- Confirm Elasticsearch is running on `http://localhost:9200`.
- If using Docker Compose, check `docker compose ps` and `docker compose logs elasticsearch`.

### No results returned

- Re-run the indexing script after checking the dataset file.
- Confirm the `utrains-qa` index exists in Elasticsearch.

## Notes for students

This project is intentionally simple and easy to read. The focus is on understanding the RAG pattern:

- retrieve relevant documents,
- add them to the prompt,
- let the model answer with those facts only.

This is a great foundation for more advanced RAG experiments such as chunking, reranking, hybrid search, and 
production-grade retrieval workflows.

# HR Policy Retrieval Chat

This project is a retrieval-only RAG application for HR policies. A separate ingestion project is responsible for cleaning, chunking, embedding, and storing policy content in Elasticsearch. This repository retrieves relevant chunks from Elasticsearch and uses them to power a Streamlit chat experience.

## Scope

This repository is responsible for:

- embedding the user query
- retrieving the most relevant chunks from Elasticsearch
- building a grounded prompt from retrieved context
- rendering the chat UI in Streamlit

This repository is not responsible for:

- document cleaning
- chunking
- embedding generation for stored documents
- index creation and bulk loading jobs

## Recommended Elasticsearch document schema

The retrieval code expects a dense vector field named `embedding`, a text field at `context.text`, and descriptive metadata at `metadata`.

```json
{
  "metadata": {
    "chunk_id": "pto_policy_2026_p03_c02",
    "document_id": "pto_policy_2026",
    "title": "Paid Time Off Policy",
    "category": "leave",
    "country": "FR",
    "version": "2026.1",
    "effective_date": "2026-01-01",
    "source": "employee_handbook.pdf",
    "page_number": 3,
    "section": "Annual Leave Entitlement",
    "chunk_index": 12
  },
  "context": {
    "text": "Employees are entitled to 25 days of paid annual leave per year..."
  },
  "embedding": [0.0123, -0.0456, 0.0789]
}
```

### Minimum recommended fields

- `metadata.chunk_id`
- `metadata.document_id`
- `metadata.title`
- `metadata.category`
- `metadata.country`
- `metadata.version`
- `metadata.effective_date`
- `metadata.source`
- `metadata.page_number`
- `metadata.section`
- `metadata.chunk_index`
- `context.text`
- `embedding`

### Strongly recommended extensions

- `metadata.language`
- `metadata.updated_at` or `metadata.ingested_at`

This schema is a good generalized baseline for HR-policy retrieval. It stays broad enough for leave, compensation, travel, benefits, conduct, and country-specific policy documents while remaining simple for retrieval and UI display.

## Project structure

```text
rag-capstone-retrieval/
├── Dockerfile
├── pyproject.toml
├── .env.example
├── README.md
├── app.py
└── src/
    ├── __init__.py
    ├── config.py
    ├── llm_client.py
    └── rag.py
```

## Setup

### 1. Install dependencies

If you do not have `uv` installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install project dependencies:

```bash
uv sync
```

### 2. Configure environment

Create a `.env` file from `.env.example` and set the required values:

```bash
cp .env.example .env
```

Example:

```env
OPENAI_API_KEY=your_openai_api_key_here
ELASTICSEARCH_HOST=http://elasticsearch:9200
INDEX_NAME=hr-policy-chunks
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
VECTOR_DIMENSION=1536
VECTOR_FIELD=embedding
STREAMLIT_PORT=8501
```

### 3. Connect to an external Elasticsearch

Provision Elasticsearch separately. The retriever container and the Elasticsearch container must join the **same Docker network** so the retriever can reach Elasticsearch by container hostname.

Example:

```bash
docker network create hr-rag-network
docker run -d --name elasticsearch --network hr-rag-network -p 9200:9200 docker.elastic.co/elasticsearch/elasticsearch:9.5.2
docker run --rm -p 8501:8501 --network hr-rag-network --env-file .env hr-policy-retrieval-chat
```

In that setup, `ELASTICSEARCH_HOST=http://elasticsearch:9200`.

### 4. Run the Streamlit app

```bash
uv run streamlit run app.py
```

### 5. Build the retriever image

```bash
docker build -t hr-policy-retrieval-chat .
```

## Retrieval flow

1. The user asks a question in the Streamlit chat.
2. The app embeds the question with the configured embedding model.
3. Elasticsearch runs kNN search against the `embedding` field.
4. The top policy chunks are returned from `context.text` with their `metadata`.
5. The app sends only those retrieved chunks to the chat model.
6. The UI displays the answer and the retrieved context for traceability.

## Troubleshooting

### OpenAI API key error

- Confirm `OPENAI_API_KEY` is set in `.env`.
- Ensure the key is active and allowed for the configured models.

### Elasticsearch connection error

- Confirm Elasticsearch is running on the configured `ELASTICSEARCH_HOST`.
- Confirm the retriever container and the Elasticsearch container are attached to the same Docker network.
- If using container hostnames, confirm the hostname in `ELASTICSEARCH_HOST` matches the Elasticsearch container name or network alias.

### No results returned

- Confirm the configured `INDEX_NAME` exists.
- Confirm documents contain `context.text` and `embedding`.
- Confirm the embedding dimension stored in Elasticsearch matches `VECTOR_DIMENSION`.

### Poor retrieval quality

- Check that the ingestion project uses the same embedding model family as retrieval.
- Verify policy metadata such as `country`, `category`, and `section` are populated consistently.
- Consider adding metadata filters in the retrieval layer for country or policy family.

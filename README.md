# Healthcare Knowledge Navigator

A production-grade **Healthcare RAG (Retrieval-Augmented Generation) Assistant** that retrieves evidence from **PMC Open Access (JATS XML)** papers and generates evidence-grounded medical answers with inline citations and confidence scores.

---

## Architecture

```
+---------------------------------------------------------------------+
|                    Healthcare Knowledge Navigator                    |
+---------------------------------------------------------------------+
|                                                                      |
|   +----------+    +-----------------+    +-------------------+       |
|   |  User    |--->|  Streamlit UI   |--->|  FastAPI Backend  |       |
|   |  Query   |    |  (Frontend)     |    |  (REST API)       |       |
|   +----------+    +-----------------+    +---------+---------+       |
|                                                    |                 |
|                                          +---------v---------+       |
|                                          |  Query Processing  |       |
|                                          +---------+---------+       |
|                                                    |                 |
|                              +---------------------+-------------+   |
|                              |                                   |   |
|                    +---------v-------+   +---------v---------+   |   |
|                    | Dense Retrieval |   |   BM25 Retrieval   |   |   |
|                    |   (MedCPT)     |   |   (Sparse Index)   |   |   |
|                    +---------+-------+   +---------+---------+   |   |
|                              |                     |             |   |
|                              +---------+-----------+             |   |
|                                        |                         |   |
|                              +---------v---------+               |   |
|                              |  Reciprocal Rank   |               |   |
|                              |     Fusion (RRF)   |               |   |
|                              +---------+---------+               |   |
|                                        |                         |   |
|                              +---------v---------+               |   |
|                              |  Cross-Encoder     |               |   |
|                              |  Re-ranking (BGE)  |               |   |
|                              +---------+---------+               |   |
|                                        |                         |   |
|                              +---------v---------+               |   |
|                              | Context Assembly   |               |   |
|                              +---------+---------+               |   |
|                                        |                         |   |
|                              +---------v---------+               |   |
|                              |    Groq LLM        |               |   |
|                              | (Llama-3.3-70B)    |               |   |
|                              +---------+---------+               |   |
|                                        |                         |   |
|                              +---------v---------+               |   |
|                              |  Evidence-based    |               |   |
|                              |  Answer + Cites    |               |   |
|                              |  + Confidence      |               |   |
|                              +-------------------+               |   |
|                                                                   |   |
|   +---------------+    +---------------+                         |   |
|   |  Qdrant       |    |  PMC XML      |                         |   |
|   |  Vector DB    |    |  Data Store   |                         |   |
|   +---------------+    +---------------+                         |   |
|                                                                      |
+---------------------------------------------------------------------+
```

---

## Tech Stack

| Layer            | Technology                          |
| ---------------- | ----------------------------------- |
| **Language**     | Python 3.12+                        |
| **Backend**      | FastAPI                             |
| **Frontend**     | Streamlit                           |
| **LLM**         | Groq API - Llama-3.3-70B-Versatile  |
| **Embeddings**   | MedCPT (modular / swappable)        |
| **Vector DB**    | Qdrant (Docker)                     |
| **Search**       | Dense + BM25 Hybrid Retrieval       |
| **Re-ranking**   | BGE Reranker Large                  |
| **XML Parsing**  | lxml, BeautifulSoup4                |
| **Chunking**     | Section-aware semantic chunking     |
| **Evaluation**   | RAGAS, DeepEval                     |
| **Config**       | Pydantic Settings + .env            |
| **Container**    | Docker + Docker Compose             |
| **Pkg Manager**  | uv (preferred) / pip (fallback)     |

---

## Installation

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/Healthcare-Knowledge-Navigator.git
cd Healthcare-Knowledge-Navigator
```

### 2. Create Virtual Environment

**Using uv (recommended):**

```bash
uv venv --python 3.12
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

**Using pip:**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

**Using uv:**

```bash
uv pip install -r requirements.txt
```

**Using pip:**

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your actual credentials (e.g., GROQ_API_KEY)
```

### 5. Verify Setup

```bash
python scripts/check_environment.py
```

---

## Running Docker

### Start All Services

```bash
docker compose up -d
```

This will spin up:

| Service      | Port  | Description              |
| ------------ | ----- | ------------------------ |
| **Qdrant**   | 6333  | Vector database (REST)   |
| **Qdrant**   | 6334  | Vector database (gRPC)   |
| **FastAPI**  | 8000  | Backend API              |

### Stop Services

```bash
docker compose down
```

### View Logs

```bash
docker compose logs -f
```

---

## Running FastAPI

### Start the Backend (Development)

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Verify the API

```bash
# Root endpoint
curl http://localhost:8000/
# Expected: {"status":"running","project":"Healthcare Knowledge Navigator"}

# Health check
curl http://localhost:8000/health
# Expected: {"status":"healthy"}
```

### API Documentation

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Running Streamlit

### Start the Frontend

```bash
streamlit run frontend/app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
Healthcare-Knowledge-Navigator/
|
+-- configs/                  # Configuration and constants
|   +-- __init__.py
|   +-- settings.py           # Pydantic-based settings management
|   +-- constants.py          # Application-wide constants
|
+-- data/                     # Data directories (gitignored)
|   +-- raw_xml/              # Raw PMC JATS XML files
|   +-- parsed_json/          # Parsed document JSONs
|   +-- chunks/               # Chunked text segments
|
+-- docs/                     # Project documentation
|
+-- ingestion/                # Data ingestion pipeline
|   +-- __init__.py
|
+-- preprocessing/            # XML parsing and text cleaning
|   +-- __init__.py
|
+-- embeddings/               # Embedding model wrappers
|   +-- __init__.py
|
+-- indexing/                 # Vector DB indexing
|   +-- __init__.py
|
+-- retrieval/                # Dense + BM25 hybrid retrieval
|   +-- __init__.py
|
+-- reranking/                # Cross-encoder re-ranking
|   +-- __init__.py
|
+-- generation/               # LLM generation and prompting
|   +-- __init__.py
|
+-- evaluation/               # RAG evaluation (RAGAS / DeepEval)
|   +-- __init__.py
|
+-- backend/                  # FastAPI application
|   +-- __init__.py
|   +-- main.py               # App entrypoint and lifespan
|   +-- config.py             # Backend configuration and logging
|   +-- routes.py             # API route definitions
|
+-- frontend/                 # Streamlit UI
|   +-- app.py                # Streamlit application
|
+-- scripts/                  # Utility and setup scripts
|   +-- setup.py              # Project initialization
|   +-- check_environment.py  # Environment verification
|
+-- tests/                    # Test suite
|   +-- __init__.py
|
+-- logs/                     # Application logs (gitignored)
+-- models/                   # Downloaded model artifacts
|
+-- .env.example              # Environment variable template
+-- .gitignore                # Git ignore rules
+-- docker-compose.yml        # Docker orchestration
+-- Dockerfile                # Container build instructions
+-- requirements.txt          # Python dependencies
+-- README.md                 # This file
```

---

## Future Roadmap

| Phase | Milestone                                | Status       |
| ----- | ---------------------------------------- | ------------ |
| 0     | Project Setup and Architecture           | Complete     |
| 1     | PMC XML Ingestion and Parsing            | Next         |
| 2     | Section-Aware Semantic Chunking          | Planned      |
| 3     | MedCPT Embeddings and Qdrant Indexing    | Planned      |
| 4     | Hybrid Retrieval (Dense + BM25)          | Planned      |
| 5     | Cross-Encoder Re-ranking (BGE)           | Planned      |
| 6     | Groq LLM Generation with Citations      | Planned      |
| 7     | Streamlit UI and End-to-End Integration  | Planned      |
| 8     | RAGAS / DeepEval Evaluation Pipeline     | Planned      |
| 9     | Production Hardening and Monitoring      | Planned      |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

> **Disclaimer**: This system is designed for research and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

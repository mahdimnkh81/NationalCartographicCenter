# Hybrid Geospatial Search Engine

A hybrid search engine for geospatial data that combines traditional lexical search with semantic vector retrieval. The system is designed to improve the search experience by understanding both exact keyword matches and the semantic meaning of user queries.

This project was developed for the **National Cartographic Center (NCC)** as part of an AI-powered geospatial search system.

---

## Features

- 🔍 Hybrid Search Architecture
  - Lexical search using Elasticsearch
  - Semantic search using Sentence-BERT (SBERT)
  - FAISS vector indexing for efficient similarity search
  - Automatic merging and ranking of lexical and semantic results

- 🧠 Semantic Retrieval
  - Transformer-based sentence embeddings
  - Fine-tuned SBERT model on geospatial search data
  - Support for natural language queries

- 📊 Search Evaluation
  - User feedback collection
  - Automatic evaluation pipeline
  - Information Retrieval metrics:
    - Mean Reciprocal Rank (MRR)
    - Mean Average Precision (MAP)
    - Precision@K

- 📈 Continuous Improvement
  - Stores user interactions
  - Uses relevance feedback to generate training datasets
  - Supports iterative model fine-tuning

---

## System Architecture

```
                User Query
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
 Elasticsearch             SBERT Encoder
 (Lexical Search)          + FAISS Search
        │                         │
        └────────────┬────────────┘
                     ▼
            Result Fusion & Ranking
                     ▼
               Search Results
                     ▼
             User Feedback Logs
                     ▼
        Evaluation & Model Fine-tuning
```

---

## Tech Stack

### Backend
- Django
- Django REST Framework
- PostgreSQL

### AI / NLP
- Hugging Face Transformers
- Sentence-BERT (SBERT)
- Fine-tuning
- NumPy
- Pandas

### Search
- Elasticsearch
- FAISS

### DevOps
- Docker

---

## Project Structure

```
project/
│
├── search/                 # Search application
├── services/               # Search & embedding services
├── models/                 # Fine-tuned SBERT models
├── data/                   # FAISS indexes and datasets
├── templates/              # Django templates
├── static/                 # Static files
└── manage.py
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/hybrid-geospatial-search.git
cd hybrid-geospatial-search
```

---

## 2. Start Elasticsearch

> **Note:** Docker is used **only** for running Elasticsearch. The Django application runs directly inside a Python virtual environment.

Create a directory for Docker Compose:

```bash
mkdir -p /home/opt/docker_compose
cd /home/opt/docker_compose
```

Create a `docker-compose.yml` file and add the Elasticsearch configuration.

Start Elasticsearch:

```bash
docker compose up -d
```

---

## 3. Create a Python Virtual Environment

```bash
sudo apt update
sudo apt install python3-venv -y

python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 4. Configure Django

Update `ALLOWED_HOSTS` inside `settings.py`:

```python
ALLOWED_HOSTS = [
    "127.0.0.1",
    "localhost",
    "your-server-ip",
]
```

Apply database migrations:

```bash
python manage.py migrate
```

---

## 5. Run the Application

### Development

```bash
python manage.py runserver
```

The application will be available at:

```
http://127.0.0.1:8000
```

---

### Production (Gunicorn)

Install Gunicorn:

```bash
pip install gunicorn
```

Run the application:

```bash
gunicorn \
    --workers 4 \
    --bind 0.0.0.0:8000 \
    --timeout 14000 \
    MappingOrganization.wsgi:application
```

The application will be available at:

```
http://<server-ip>:8000
```

---

## Hybrid Search Pipeline

1. User submits a search query.
2. The query is sent to:
   - Elasticsearch for lexical retrieval.
   - SBERT for dense embedding generation.
3. FAISS retrieves the most semantically similar documents.
4. Results from both retrieval methods are combined.
5. Final ranking is returned to the user.

---

## Model Training

The semantic model is fine-tuned using user relevance feedback collected during search sessions.

Training pipeline:

- Collect search logs
- Generate query-document pairs
- Convert relevance labels into training examples
- Fine-tune SBERT
- Generate updated embeddings
- Rebuild the FAISS index

---

## Evaluation Metrics

The retrieval system is evaluated using standard Information Retrieval metrics:

- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)
- Precision@K

These metrics are automatically computed from user interaction logs and are used to monitor retrieval quality after each training cycle.

---

## Key Contributions

- Designed and implemented a hybrid retrieval architecture combining lexical and semantic search.
- Built a complete relevance feedback and evaluation pipeline.
- Fine-tuned SBERT on geospatial search data.
- Developed scalable vector search using FAISS.
- Integrated the entire system into a Django backend with REST APIs.

---

## Future Improvements

- Replace FAISS with a distributed vector database (Milvus or Qdrant).
- Introduce hybrid ranking using Reciprocal Rank Fusion (RRF).
- Support multilingual retrieval.
- Integrate Retrieval-Augmented Generation (RAG).
- Add cross-encoder reranking for improved ranking quality.

---

## License

This repository is provided for research and portfolio purposes.

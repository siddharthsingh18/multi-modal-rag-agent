# Multi-Modal RAG Agent with Agentic Workflows

A production-grade Retrieval-Augmented Generation (RAG) system with advanced agentic capabilities, built with LangGraph, Anthropic Claude, and modern Python stack.

## 🎯 Features

- **Multi-Modal Document Processing**: PDF, images (OCR), tables, markdown, HTML, and code
- **Hybrid Search**: Dense (semantic) + sparse (BM25) retrieval with re-ranking
- **Agentic Workflows**: LangGraph-powered state machine with planning, tool use, and self-reflection
- **Production-Ready API**: FastAPI with async endpoints, rate limiting, and CORS
- **Advanced Retrieval**: Query transformation, HyDE, multi-query generation
- **Full Observability**: LangSmith tracing, Prometheus metrics, structured logging
- **Evaluation Framework**: RAGAS metrics and quality benchmarking
- **Scalable Architecture**: Redis caching, vector DB (Qdrant/ChromaDB), Docker deployment

## 📋 Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI REST API                          │
│  ┌──────────┬──────────┬──────────┬──────────┬────────────┐    │
│  │  Query   │  Ingest  │  Health  │  Metrics │  Streaming │    │
│  └──────────┴──────────┴──────────┴──────────┴────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Agent (LangGraph)                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Planner → Retriever → Re-Ranker → Generator → Reflector │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
           │                  │                  │
           ▼                  ▼                  ▼
┌────────────────┐  ┌──────────────────┐  ┌────────────────┐
│  Tool Registry │  │  Hybrid Search   │  │  LLM Client    │
│  - Retriever   │  │  - Dense (Vec)   │  │  (Claude 3.5)  │
│  - Calculator  │  │  - Sparse (BM25) │  └────────────────┘
│  - Web Search  │  │  - Re-ranking    │
│  - Code Exec   │  └──────────────────┘
└────────────────┘           │
                             ▼
                  ┌───────────────────────┐
                  │  Vector Database      │
                  │  - Qdrant / ChromaDB  │
                  └───────────────────────┘
```

## 🚀 Quick Start

### 5-Minute Setup

1. **Clone and navigate:**
```bash
git clone <repository-url>
cd multimodal-rag-agent
```

2. **Create environment file:**
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

3. **Start with Docker:**
```bash
docker-compose -f docker/docker-compose.yml up -d
```

4. **Verify it's running:**
```bash
curl http://localhost:8000/api/v1/health
```

5. **Test query:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

## 📦 Installation

### Prerequisites

- Python 3.10+
- Python 3.10, 3.11, or 3.12 is recommended (Python 3.14 is not supported)
- Docker & Docker Compose (for containerized deployment)
- Tesseract OCR (for image processing)
- Anthropic API key

### Local Development Setup

1. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
python -m pip install -e ".[dev]"
```

3. **Install Tesseract (for OCR):**
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

4. **Set up environment:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Start services:**
```bash
# Option 1: Use Docker for dependencies only
docker-compose -f docker/docker-compose.yml up -d qdrant redis

# Option 2: Install locally (Qdrant, Redis)
```

## ⚙️ Configuration

### Environment Variables

Edit `.env` file with your configuration:

```bash
# LLM Configuration
ANTHROPIC_API_KEY=your_api_key_here
LLM_MODEL=claude-3-5-sonnet-20241022
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=4096

# Vector Database (choose one)
QDRANT_URL=http://localhost:6333
# OR
CHROMA_PERSIST_DIR=./data/chroma
VECTOR_DB_TYPE=qdrant  # or 'chroma'

# Redis Cache
REDIS_URL=redis://localhost:6379/0

# API authentication (required in production)
API_AUTH_KEY=replace_with_a_long_random_value

# Observability
LANGSMITH_API_KEY=your_langsmith_key  # Optional
LANGSMITH_TRACING=false
LOG_LEVEL=INFO

# Retrieval Configuration
RETRIEVAL_TOP_K=10
RERANK_TOP_K=5
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

## 📚 Usage

### Starting the Application

**Development mode:**
```bash
make run
# or
uvicorn src.api.main:app --reload
```

**Production mode:**
```bash
make run-prod
# or
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Document Ingestion

**Ingest a directory:**
```bash
python scripts/ingest_documents.py /path/to/documents --collection my_docs
```

**Via API:**
```bash
# Single document
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your document text here",
    "metadata": {"source": "api", "type": "text"}
  }'

# Batch from directory
curl -X POST http://localhost:8000/api/v1/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{
    "directory": "/path/to/documents",
    "recursive": true
  }'

# File upload
curl -X POST http://localhost:8000/api/v1/ingest/upload \
  -F "file=@document.pdf"
```

### Querying the RAG System

**Python SDK:**
```python
import asyncio
from src.agents.rag_agent import create_rag_agent
from src.retrieval.vector_store import get_retriever

async def query_rag():
    retriever = await get_retriever()
    agent = await create_rag_agent(retriever)

    result = await agent.run(
        query="What is machine learning?",
    )

    print(f"Answer: {result['answer']}")
    print(f"Documents: {len(result['documents'])}")
    print(f"Reflection Score: {result['reflection']['score']}/10")

asyncio.run(query_rag())
```

**API Call:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_AUTH_KEY}" \
  -d '{
    "query": "Explain neural networks",
    "top_k": 5,
    "use_reflection": true
  }'
```

**Streaming Response:**
```bash
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_AUTH_KEY}" \
  -d '{"query": "What is deep learning?"}' \
  --no-buffer
```

## 📖 API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/query` | POST | Query the RAG system |
| `/api/v1/query/stream` | POST | Stream query results |
| `/api/v1/ingest` | POST | Ingest single document |
| `/api/v1/ingest/batch` | POST | Batch ingest from directory |
| `/api/v1/ingest/upload` | POST | Upload and ingest file |
| `/api/v1/health` | GET | Health check with dependencies |
| `/api/v1/health/readiness` | GET | Kubernetes readiness probe |
| `/api/v1/health/liveness` | GET | Kubernetes liveness probe |

## 🛠️ Development

### Project Structure

```
multimodal-rag-agent/
├── src/
│   ├── agents/           # Agent workflows and tools
│   ├── api/              # FastAPI application
│   ├── database/         # Database models and interfaces
│   ├── evaluation/       # Metrics and benchmarking
│   ├── generation/       # LLM client and prompts
│   ├── ingestion/        # Document loaders and chunking
│   ├── observability/    # Logging, tracing, metrics
│   ├── retrieval/        # Embeddings, vector store, search
│   └── utils/            # Config, cache, exceptions
├── tests/                # Unit, integration, evaluation tests
├── scripts/              # CLI scripts for operations
├── configs/              # Configuration files
└── docker/               # Docker and deployment files
```

### Code Quality

**Linting and formatting:**
```bash
make lint    # Check code quality
make format  # Auto-fix issues
```

**Type checking:**
```bash
# Ruff includes basic type checking
make lint
```

## 🧪 Testing

**Run all tests:**
```bash
make test
```

**Run with coverage:**
```bash
make test-cov
```

**Run specific test suites:**
```bash
make test-unit         # Unit tests only
make test-integration  # Integration tests only
```

**Individual test file:**
```bash
pytest tests/unit/test_retrieval.py -v
```

### Evaluation

**Run quality evaluation:**
```bash
python scripts/evaluate_rag.py configs/evaluation/sample_queries.json
```

**Run performance benchmark:**
```bash
python scripts/benchmark.py --num-queries 20
```

## 🚢 Deployment

### Docker Deployment

**Build image:**
```bash
make docker-build
```

**Start all services:**
```bash
make docker-up
```

**View logs:**
```bash
make docker-logs
```

**Stop services:**
```bash
make docker-down
```

### Production Checklist

- [ ] Set `ENVIRONMENT=production` in `.env`
- [ ] Configure proper `ANTHROPIC_API_KEY`
- [ ] Set up persistent volumes for data
- [ ] Configure CORS origins
- [ ] Set up Prometheus and Grafana
- [ ] Enable LangSmith tracing
- [ ] Configure rate limiting
- [ ] Set up health checks
- [ ] Configure SSL/TLS
- [ ] Set up log aggregation

### Kubernetes Deployment

Example Kubernetes manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-agent
  template:
    metadata:
      labels:
        app: rag-agent
    spec:
      containers:
      - name: rag-agent
        image: multimodal-rag-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: anthropic-api-key
        livenessProbe:
          httpGet:
            path: /api/v1/health/liveness
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health/readiness
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

## 🔍 Troubleshooting

### Common Issues

**1. "ANTHROPIC_API_KEY not found"**
```bash
# Ensure .env file exists and contains your API key
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY=your_key
```

**2. "Connection refused to Qdrant/Redis"**
```bash
# Start dependencies with Docker
docker-compose -f docker/docker-compose.yml up -d qdrant redis

# Or check if services are running
docker ps
```

**3. "Module not found" errors**
```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**4. OCR not working**
```bash
# Install Tesseract OCR
brew install tesseract  # macOS
sudo apt-get install tesseract-ocr  # Ubuntu
```

**5. Out of memory errors**
```bash
# Reduce batch size in configuration
CHUNK_SIZE=256
RETRIEVAL_TOP_K=5
```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python -m src.api.main
```

### Health Check

```bash
# Check system health
curl http://localhost:8000/api/v1/health

# Expected response:
{
  "status": "healthy",
  "version": "0.1.0",
  "dependencies": {
    "llm": "configured",
    "vector_db": "healthy",
    "redis": "healthy"
  }
}
```

## 📊 Monitoring

### Prometheus Metrics

Available at: `http://localhost:9090`

Key metrics:
- `rag_queries_total` - Total RAG queries
- `rag_query_duration_seconds` - Query latency
- `rag_reflection_score` - Answer quality scores
- `llm_requests_total` - LLM API calls
- `cache_hits_total` / `cache_misses_total` - Cache performance

### Grafana Dashboards

Available at: `http://localhost:3000` (admin/admin)

Import dashboards for:
- RAG query performance
- LLM usage and costs
- System resource utilization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - LLM framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agentic workflows
- [Anthropic](https://anthropic.com/) - Claude LLM
- [Qdrant](https://qdrant.tech/) - Vector database
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework

## 📞 Support

- GitHub Issues: [Create an issue](https://github.com/yourusername/multimodal-rag-agent/issues)
- Documentation: See `/docs` endpoint when running
- Email: support@example.com

---

**Built with ❤️ using Python, LangGraph, and Claude**

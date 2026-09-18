.PHONY: install test run lint format clean docker-up docker-down ingest evaluate

# Installation
install:
	python -m pip install -e .

install-dev: install
	python -m pip install -e ".[dev]"

# Development
run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Code Quality
lint:
	ruff check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# Docker
docker-build:
	docker build -f docker/Dockerfile -t multimodal-rag-agent .

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f

# Scripts
ingest:
	python scripts/ingest_documents.py

evaluate:
	python scripts/evaluate_rag.py

benchmark:
	python scripts/benchmark.py

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

# Database
db-init:
	alembic init alembic

db-migrate:
	alembic revision --autogenerate -m "migration"

db-upgrade:
	alembic upgrade head

# Help
help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make run           - Run development server"
	@echo "  make test          - Run all tests"
	@echo "  make lint          - Lint code"
	@echo "  make format        - Format code"
	@echo "  make docker-up     - Start Docker services"
	@echo "  make docker-down   - Stop Docker services"
	@echo "  make ingest        - Run document ingestion"
	@echo "  make evaluate      - Run RAG evaluation"
	@echo "  make clean         - Clean build artifacts"

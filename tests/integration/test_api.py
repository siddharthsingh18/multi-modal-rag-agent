"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import (
    get_rag_agent_dependency,
    get_retriever_dependency,
    get_settings_dependency,
    verify_api_key,
)
from src.api.main import app
from src.utils.config import Settings

client = TestClient(app)


class FakeAgent:
    async def run(self, **kwargs):
        return {
            "query": kwargs["query"],
            "answer": "Python is a programming language.",
            "documents": [
                {"id": "doc-1", "score": 0.9, "content": "Python is a language."}
            ],
            "reflection": {"score": 8},
            "metadata": {"num_documents": 1},
        }


class FakeRetriever:
    async def add_texts(self, texts, metadata=None, ids=None):
        return ids or ["generated-id"]


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "dependencies" in data

    def test_readiness(self):
        """Test readiness probe."""
        response = client.get("/api/v1/health/readiness")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True

    def test_liveness(self):
        """Test liveness probe."""
        response = client.get("/api/v1/health/liveness")
        assert response.status_code == 200
        data = response.json()
        assert data["alive"] is True


class TestQueryEndpoints:
    """Tests for query endpoints."""

    def test_query_requires_api_key_when_configured(self):
        """Reject protected requests without the configured API key."""
        settings = Settings(
            _env_file=None,
            anthropic_api_key="test-key",
            api_auth_key="expected-key",
        )
        app.dependency_overrides[get_settings_dependency] = lambda: settings

        response = client.post("/api/v1/query", json={"query": "What is Python?"})

        assert response.status_code == 401

    def test_query_endpoint(self):
        """Return a serialized answer from a mocked agent."""
        app.dependency_overrides[get_rag_agent_dependency] = lambda: FakeAgent()

        response = client.post(
            "/api/v1/query",
            headers={"X-API-Key": "development"},
            json={"query": "What is Python?", "top_k": 1},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Python is a programming language."
        assert data["documents"][0]["id"] == "doc-1"


class TestIngestEndpoints:
    """Tests for ingestion endpoints."""

    def test_ingest_text(self):
        """Ingest text through the route using a mocked retriever."""
        request_data = {
            "text": "This is a test document for ingestion.",
            "metadata": {"source": "test", "type": "text"},
        }
        app.dependency_overrides[get_retriever_dependency] = lambda: FakeRetriever()

        response = client.post(
            "/api/v1/ingest",
            headers={"X-API-Key": "development"},
            json=request_data,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["num_documents"] == 1
        assert data["num_chunks"] == 1

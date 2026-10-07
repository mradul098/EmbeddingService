"""
Tests for the embedding endpoint and service layer.

These tests mock the Vertex AI SDK so they can run without GCP credentials.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    from app.main import app

    return TestClient(app)


def _make_mock_embedding(values: list[float], token_count: int) -> MagicMock:
    """Helper: create a mock Vertex AI embedding result."""
    emb = MagicMock()
    emb.values = values
    emb.statistics.token_count = token_count
    return emb


# ---------------------------------------------------------------------------
# POST /embed — happy-path tests
# ---------------------------------------------------------------------------

class TestEmbedEndpointSuccess:
    """Successful embedding requests."""

    @patch(
        "app.services.embedding_provider_client.TextEmbeddingModel.from_pretrained"
    )
    def test_single_text_input(self, mock_from_pretrained, client):
        """Single string input returns one embedding vector."""
        mock_model = MagicMock()
        mock_model.get_embeddings.return_value = [
            _make_mock_embedding([0.1, 0.2, 0.3], token_count=5),
        ]
        mock_from_pretrained.return_value = mock_model

        response = client.post(
            "/embed",
            json={"input": "Hello world"},
        )

        assert response.status_code == 200
        body = response.json()
        assert len(body["data"]) == 1
        assert body["data"][0]["index"] == 0
        assert body["data"][0]["embedding"] == [0.1, 0.2, 0.3]
        assert body["usage"]["prompt_tokens"] == 5
        assert body["usage"]["total_tokens"] == 5

    @patch(
        "app.services.embedding_provider_client.TextEmbeddingModel.from_pretrained"
    )
    def test_batch_text_input(self, mock_from_pretrained, client):
        """List of strings returns one embedding per input."""
        mock_model = MagicMock()
        mock_model.get_embeddings.return_value = [
            _make_mock_embedding([0.1, 0.2], token_count=3),
            _make_mock_embedding([0.4, 0.5], token_count=4),
        ]
        mock_from_pretrained.return_value = mock_model

        response = client.post(
            "/embed",
            json={"input": ["First text", "Second text"]},
        )

        assert response.status_code == 200
        body = response.json()
        assert len(body["data"]) == 2
        assert body["data"][0]["index"] == 0
        assert body["data"][1]["index"] == 1
        assert body["usage"]["total_tokens"] == 7

    @patch(
        "app.services.embedding_provider_client.TextEmbeddingModel.from_pretrained"
    )
    def test_with_task_type(self, mock_from_pretrained, client):
        """Task type is forwarded to the provider."""
        mock_model = MagicMock()
        mock_model.get_embeddings.return_value = [
            _make_mock_embedding([0.1], token_count=2),
        ]
        mock_from_pretrained.return_value = mock_model

        response = client.post(
            "/embed",
            json={
                "input": "query text",
                "task_type": "RETRIEVAL_QUERY",
            },
        )

        assert response.status_code == 200
        # Verify TextEmbeddingInput was created with the task_type
        call_args = mock_model.get_embeddings.call_args
        inputs = call_args[0][0]
        assert inputs[0].task_type == "RETRIEVAL_QUERY"

    @patch(
        "app.services.embedding_provider_client.TextEmbeddingModel.from_pretrained"
    )
    def test_with_output_dimensionality(self, mock_from_pretrained, client):
        """Output dimensionality is passed to the SDK."""
        mock_model = MagicMock()
        mock_model.get_embeddings.return_value = [
            _make_mock_embedding([0.1, 0.2], token_count=3),
        ]
        mock_from_pretrained.return_value = mock_model

        response = client.post(
            "/embed",
            json={
                "input": "some text",
                "output_dimensionality": 256,
            },
        )

        assert response.status_code == 200
        call_kwargs = mock_model.get_embeddings.call_args[1]
        assert call_kwargs["output_dimensionality"] == 256

    @patch(
        "app.services.embedding_provider_client.TextEmbeddingModel.from_pretrained"
    )
    def test_response_includes_model_name(self, mock_from_pretrained, client):
        """Response contains the model that was actually used."""
        mock_model = MagicMock()
        mock_model.get_embeddings.return_value = [
            _make_mock_embedding([0.1], token_count=1),
        ]
        mock_from_pretrained.return_value = mock_model

        response = client.post(
            "/embed",
            json={
                "input": "text",
                "model": "text-embedding-005",
            },
        )

        assert response.status_code == 200
        assert response.json()["model"] == "text-embedding-005"


# ---------------------------------------------------------------------------
# POST /embed — validation & error tests
# ---------------------------------------------------------------------------

class TestEmbedEndpointErrors:
    """Requests that should fail with 4xx errors."""

    def test_empty_string_input(self, client):
        """Empty string should be rejected by validation."""
        response = client.post("/embed", json={"input": ""})
        assert response.status_code == 422

    def test_empty_list_input(self, client):
        """Empty list should be rejected by validation."""
        response = client.post("/embed", json={"input": []})
        assert response.status_code == 422

    def test_list_with_empty_string(self, client):
        """List containing an empty string should be rejected."""
        response = client.post("/embed", json={"input": ["valid", ""]})
        assert response.status_code == 422

    def test_missing_input_field(self, client):
        """Missing required 'input' field should return 422."""
        response = client.post("/embed", json={})
        assert response.status_code == 422

    def test_invalid_task_type(self, client):
        """Invalid task_type should be rejected by validation."""
        response = client.post(
            "/embed",
            json={"input": "text", "task_type": "INVALID_TASK"},
        )
        assert response.status_code == 422

    def test_unsupported_model(self, client):
        """Model not in the supported list should return 400."""
        response = client.post(
            "/embed",
            json={"input": "text", "model": "nonexistent-model-xyz"},
        )
        assert response.status_code == 400
        assert "not supported" in response.json()["detail"].lower()

    @patch(
        "app.services.embedding_provider_client.TextEmbeddingModel.from_pretrained"
    )
    def test_batch_exceeds_max_size(self, mock_from_pretrained, client):
        """Batch larger than MAX_EMBEDDING_BATCH_SIZE should return 400."""
        # Create a batch that exceeds the limit (default 250)
        oversized_batch = [f"text_{i}" for i in range(251)]
        response = client.post(
            "/embed",
            json={"input": oversized_batch},
        )
        assert response.status_code == 400
        assert "exceeds maximum" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# POST /embed — provider error handling
# ---------------------------------------------------------------------------

class TestEmbedProviderErrors:
    """Vertex AI SDK errors are handled gracefully."""

    @patch(
        "app.services.embedding_provider_client.TextEmbeddingModel.from_pretrained"
    )
    def test_provider_exception_returns_500(self, mock_from_pretrained, client):
        """Unhandled provider errors should return 500."""
        mock_model = MagicMock()
        mock_model.get_embeddings.side_effect = RuntimeError("Vertex AI is down")
        mock_from_pretrained.return_value = mock_model

        response = client.post(
            "/embed",
            json={"input": "some text"},
        )

        assert response.status_code == 500
        assert "internal error" in response.json()["detail"].lower()

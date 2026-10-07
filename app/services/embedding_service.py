"""
Service layer for embedding operations.

Orchestrates validation, provider calls, usage logging,
and response construction for the /embed endpoint.
"""

from typing import List, Union

from app.core.config import settings
from app.models.embedding_schemas import (
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from app.resilience.exceptions import (
    EmbeddingBatchTooLargeError,
    EmbeddingModelNotSupportedError,
)
from app.services.embedding_provider_client import embedding_provider_client
from app.services.usage_logger import log_embedding_usage
from app.utilities.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Handles embedding request orchestration."""

    def __init__(self) -> None:
        self.provider_client = embedding_provider_client

    def _normalize_input(self, input_data: Union[str, List[str]]) -> List[str]:
        """Ensure input is always a list of strings."""
        if isinstance(input_data, str):
            return [input_data]
        return input_data

    def _resolve_model(self, model: str | None) -> str:
        """Resolve the model name, falling back to the configured default."""
        return model or settings.DEFAULT_EMBEDDING_MODEL

    def _validate_model(self, model: str) -> None:
        """Check that the requested model is in the supported list."""
        if model not in settings.SUPPORTED_EMBEDDING_MODELS:
            raise EmbeddingModelNotSupportedError(
                f"Model '{model}' is not supported. "
                f"Supported models: {settings.SUPPORTED_EMBEDDING_MODELS}"
            )

    def _validate_batch_size(self, texts: List[str]) -> None:
        """Ensure the batch does not exceed the configured maximum."""
        if len(texts) > settings.MAX_EMBEDDING_BATCH_SIZE:
            raise EmbeddingBatchTooLargeError(
                f"Batch size {len(texts)} exceeds maximum of "
                f"{settings.MAX_EMBEDDING_BATCH_SIZE}."
            )

    async def create_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Process an embedding request end-to-end.

        Steps:
            1. Normalize input to a list of strings.
            2. Resolve and validate the model name.
            3. Validate batch size.
            4. Call the Vertex AI provider client.
            5. Build the response.
            6. Log usage.

        Args:
            request: The validated EmbeddingRequest.

        Returns:
            EmbeddingResponse with embedding vectors and usage info.

        Raises:
            EmbeddingModelNotSupportedError: If the model isn't in the allow-list.
            EmbeddingBatchTooLargeError: If batch size exceeds the limit.
        """
        # 1. Normalize
        texts = self._normalize_input(request.input)

        # 2. Resolve & validate model
        model_name = self._resolve_model(request.model)
        self._validate_model(model_name)

        # 3. Validate batch size
        self._validate_batch_size(texts)

        logger.info(
            f"Processing embedding request: model={model_name}, "
            f"num_texts={len(texts)}, task_type={request.task_type}"
        )

        # 4. Call provider
        vectors, total_tokens, latency_ms = self.provider_client.get_embeddings(
            texts=texts,
            model_name=model_name,
            task_type=request.task_type,
            title=request.title,
            output_dimensionality=request.output_dimensionality,
            auto_truncate=request.auto_truncate,
        )

        # 5. Build response
        embedding_objects = [
            EmbeddingObject(index=i, embedding=vector)
            for i, vector in enumerate(vectors)
        ]
        usage = EmbeddingUsage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens,
        )
        response = EmbeddingResponse(
            model=model_name,
            data=embedding_objects,
            usage=usage,
        )

        # 6. Log usage
        log_embedding_usage(
            model=model_name,
            num_inputs=len(texts),
            total_tokens=total_tokens,
            latency_ms=latency_ms,
        )

        logger.info(
            f"Embedding request completed: model={model_name}, "
            f"num_vectors={len(vectors)}, total_tokens={total_tokens}, "
            f"latency_ms={latency_ms:.1f}"
        )

        return response


# Module-level singleton
embedding_service = EmbeddingService()

"""
Provider client for Vertex AI embedding models.

Handles direct communication with the GCP Vertex AI TextEmbeddingModel SDK.
"""

import time
from typing import Dict, List, Optional, Tuple

from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from app.core.config import settings
from app.utilities.logger import get_logger

logger = get_logger(__name__)


class EmbeddingProviderClient:
    """Client that wraps Vertex AI SDK calls for text embeddings."""

    def __init__(self) -> None:
        self._model_cache: Dict[str, TextEmbeddingModel] = {}

    def _get_model(self, model_name: str) -> TextEmbeddingModel:
        """Load and cache the embedding model instance."""
        if model_name not in self._model_cache:
            logger.info(f"Loading embedding model: {model_name}")
            self._model_cache[model_name] = TextEmbeddingModel.from_pretrained(
                model_name
            )
        return self._model_cache[model_name]

    def get_embeddings(
        self,
        texts: List[str],
        model_name: str,
        task_type: Optional[str] = None,
        title: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
        auto_truncate: bool = True,
    ) -> Tuple[List[List[float]], int, float]:
        """
        Generate embeddings for a list of texts using Vertex AI.

        Args:
            texts: List of text strings to embed.
            model_name: The Vertex AI embedding model identifier.
            task_type: Optional downstream task hint for the model.
            title: Optional title (only used with RETRIEVAL_DOCUMENT task_type).
            output_dimensionality: Optional dimension to truncate embeddings to.
            auto_truncate: Whether to auto-truncate inputs exceeding the token limit.

        Returns:
            A tuple of:
                - List of embedding vectors (List[List[float]])
                - Total token count (int)
                - Latency in milliseconds (float)

        Raises:
            Exception: Propagates any Vertex AI SDK errors.
        """
        model = self._get_model(model_name)

        # Build TextEmbeddingInput objects
        inputs = []
        for text in texts:
            if task_type:
                inputs.append(TextEmbeddingInput(text=text, task_type=task_type, title=title))
            else:
                inputs.append(TextEmbeddingInput(text=text, title=title))

        # Build kwargs for the SDK call
        kwargs = {}
        if output_dimensionality is not None:
            kwargs["output_dimensionality"] = output_dimensionality
        if not auto_truncate:
            kwargs["auto_truncate"] = False

        logger.info(
            f"Requesting embeddings: model={model_name}, "
            f"num_inputs={len(texts)}, task_type={task_type}, "
            f"output_dimensionality={output_dimensionality}"
        )

        start_time = time.time()
        embeddings = model.get_embeddings(inputs, **kwargs)
        latency_ms = (time.time() - start_time) * 1000

        # Extract vectors and token counts
        vectors = [embedding.values for embedding in embeddings]
        total_tokens = sum(
            embedding.statistics.token_count for embedding in embeddings
        )

        logger.info(
            f"Embeddings received: num_vectors={len(vectors)}, "
            f"total_tokens={total_tokens}, latency_ms={latency_ms:.1f}"
        )

        return vectors, total_tokens, latency_ms


# Module-level singleton for reuse across requests
embedding_provider_client = EmbeddingProviderClient()

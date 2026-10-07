"""
API router for the embedding endpoint.
"""

from fastapi import APIRouter, HTTPException

from app.models.embedding_schemas import EmbeddingRequest, EmbeddingResponse
from app.resilience.exceptions import (
    EmbeddingBatchTooLargeError,
    EmbeddingModelNotSupportedError,
)
from app.services.embedding_service import embedding_service
from app.utilities.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Embeddings"])


@router.post("/embed", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Generate embeddings for one or more text inputs using Vertex AI.

    Accepts a single string or a list of strings and returns
    an embedding vector for each input.
    """
    try:
        response = await embedding_service.create_embedding(request)
        return response

    except EmbeddingModelNotSupportedError as e:
        logger.warning(f"Unsupported embedding model requested: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except EmbeddingBatchTooLargeError as e:
        logger.warning(f"Embedding batch too large: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Embedding request failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while generating embeddings.",
        )

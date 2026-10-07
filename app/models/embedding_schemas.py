"""
Pydantic schemas for the embedding endpoint.
"""

from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class EmbeddingRequest(BaseModel):
    """Request schema for the /embed endpoint."""

    input: Union[str, List[str]] = Field(
        ...,
        description="A single text string or a list of text strings to embed.",
    )
    model: Optional[str] = Field(
        default=None,
        description=(
            "The embedding model to use. "
            "Defaults to the server-configured default if not specified."
        ),
    )
    task_type: Optional[str] = Field(
        default=None,
        description=(
            "The intended downstream task. Helps the model produce higher-quality "
            "embeddings. Supported values: RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, "
            "SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING, "
            "QUESTION_ANSWERING, FACT_VERIFICATION."
        ),
    )
    title: Optional[str] = Field(
        default=None,
        description=(
            "An optional title for the text. Only applicable when "
            "task_type is RETRIEVAL_DOCUMENT."
        ),
    )
    output_dimensionality: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Desired output embedding dimension. The embedding vector will be "
            "truncated to this size. If not set, the model's default is used (typically 768)."
        ),
    )
    auto_truncate: Optional[bool] = Field(
        default=True,
        description=(
            "If True, inputs that exceed the model's token limit are automatically "
            "truncated. If False, oversized inputs return an error."
        ),
    )

    @field_validator("input")
    @classmethod
    def input_must_not_be_empty(cls, v: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Input text must not be empty.")
        elif isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Input list must contain at least one text string.")
            for i, text in enumerate(v):
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"Input at index {i} must be a non-empty string."
                    )
        return v

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {
            "RETRIEVAL_DOCUMENT",
            "RETRIEVAL_QUERY",
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
            "QUESTION_ANSWERING",
            "FACT_VERIFICATION",
        }
        if v not in allowed:
            raise ValueError(
                f"Invalid task_type '{v}'. Must be one of: {', '.join(sorted(allowed))}"
            )
        return v


class EmbeddingObject(BaseModel):
    """A single embedding result."""

    index: int = Field(..., description="Index of this embedding in the input list.")
    embedding: List[float] = Field(..., description="The embedding vector.")


class EmbeddingUsage(BaseModel):
    """Token usage statistics for the embedding request."""

    prompt_tokens: int = Field(
        ..., description="Total tokens consumed across all inputs."
    )
    total_tokens: int = Field(
        ..., description="Total tokens used (same as prompt_tokens for embeddings)."
    )


class EmbeddingResponse(BaseModel):
    """Response schema for the /embed endpoint."""

    model: str = Field(..., description="The embedding model that was used.")
    data: List[EmbeddingObject] = Field(
        ..., description="List of embedding results, one per input text."
    )
    usage: EmbeddingUsage = Field(..., description="Token usage statistics.")

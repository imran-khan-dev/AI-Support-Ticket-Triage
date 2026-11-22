from pydantic import BaseModel, Field, field_validator
from typing import Any, Literal


def strip_titles(schema_obj: Any) -> Any:
    """Recursively remove ``title`` keys from a JSON schema dict.

    Pydantic's ``model_json_schema()`` adds ``title`` fields at the root and
    on every property. OpenAI's function tool parameters do not need them and
    some providers reject unknown schema fields.
    """
    if isinstance(schema_obj, dict):
        return {
            key: strip_titles(value)
            for key, value in schema_obj.items()
            if key != "title"
        }
    if isinstance(schema_obj, list):
        return [strip_titles(item) for item in schema_obj]
    return schema_obj


class SearchKnowledgeArgs(BaseModel):
    """Arguments for searching the tenant's knowledge base."""
    query: str = Field(
        ...,
        description="The search query to match against the tenant's support documentation.",
        min_length=1,
        max_length=500,
        examples=["password reset", "billing refund", "login troubleshooting"],
    )
    top_k: int = Field(
        default=5,
        description="Maximum number of results to return.",
        ge=1,
        le=50,
    )
    mode: Literal["bm25", "dense", "hybrid"] = Field(
        default="bm25",
        description="Retrieval mode: 'bm25' for keyword search, 'dense' for semantic vector search, 'hybrid' for combined BM25 and dense with RRF.",
    )
    rerank: bool = Field(
        default=False,
        description="Whether to apply cross-encoder reranking to improve relevance ordering. Requires reranking to be enabled in the knowledge service.",
    )

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value


class SearchSupportDocsArgs(BaseModel):
    """Arguments for searching support documentation."""

    query: str = Field(
        ...,
        description="The search query to match against support documentation.",
        min_length=1,
        max_length=500,
        examples=["password reset", "billing refund", "login troubleshooting"],
    )

    @field_validator('query', mode='before')
    @classmethod
    def strip_query(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value


class LookupTicketHistoryArgs(BaseModel):
    """Arguments for looking up ticket history."""

    query: str = Field(
        ...,
        description="The search query to match against ticket history records.",
        min_length=1,
        max_length=500,
        examples=["password reset", "billing refund", "login issue"],
    )

    @field_validator('query', mode='before')
    @classmethod
    def strip_query(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value


class CreateSupportCaseArgs(BaseModel):
    """Arguments for creating a support case."""

    subject: str = Field(
        ...,
        description="Short summary of the support case.",
        min_length=1,
        max_length=200,
    )
    description: str = Field(
        ...,
        description="Detailed description of the issue.",
        min_length=1,
        max_length=2000,
    )
    priority: str = Field(
        ...,
        description="Case priority level.",
        pattern=r"^(low|medium|high|urgent)$",
    )
    category: str = Field(
        ...,
        description="Support case category.",
        min_length=1,
        max_length=100,
    )

    @field_validator('subject', mode='before')
    @classmethod
    def strip_subject(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator('description', mode='before')
    @classmethod
    def strip_description(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator('category', mode='before')
    @classmethod
    def strip_category(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value

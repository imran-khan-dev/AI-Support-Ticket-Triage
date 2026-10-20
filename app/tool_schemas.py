from pydantic import BaseModel, Field, field_validator


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

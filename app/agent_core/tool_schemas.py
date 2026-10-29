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

from pydantic import BaseModel


class LLMResult(BaseModel):
    success: bool
    content: str | None = None
    error: str | None = None
    model_used: str
    tokens_used: int = 0

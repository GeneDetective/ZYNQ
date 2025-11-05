from pydantic import BaseModel

from zynq.models.prompt import BasePrompt


class BaseLLMProviderResponse(BaseModel):
    response: str


class BaseLLMProviderRequest(BaseModel):
    prompt: BasePrompt

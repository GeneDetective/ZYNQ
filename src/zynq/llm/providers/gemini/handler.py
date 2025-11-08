# src/zynq/llm/providers/gemini/handler.py
import logging
import os
from typing import Any, Optional, Union

import aiohttp
import backoff

from zynq.enums import EnvironmentVariables, LLMRole
from zynq.llm.models import BaseLLMProviderResponse
from zynq.llm.providers.base import (
    BaseLLMMessage,
    BaseLLMProvider,
    BaseLLMProviderException,
    BaseLLMProviderRateLimitException,
    llm_provider_fm,
)
from zynq.llm.providers.enums import LLMProvider
from zynq.llm.providers.gemini.models import GenerateContentRequest, GenerateContentResponse, SafetySetting
from zynq.llm.providers.shared.decorators import api_endpoint

logger = logging.getLogger(__name__)


class GeminiProviderException(BaseLLMProviderException):
    pass


GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1/models/"


@llm_provider_fm.flavor(LLMProvider.GEMINI)
class GeminiProvider(BaseLLMProvider):
    """
    Gemini provider flavor.

    Notes:
      - Do not instantiate aiohttp.ClientSession synchronously; create lazily in async context.
      - generate() and chat() accept `url` optionally and will construct a sensible default
        from the provider's stored base URL / model name when not provided.
    """

    def __init__(self, model: str, safety_settings: Optional[list[SafetySetting]] = None, **extra: Any):
        super().__init__(model=model, **extra)

        if (api_key := os.environ.get(EnvironmentVariables.GEMINI_API_KEY.value)) is None:
            raise GeminiProviderException(f"{EnvironmentVariables.GEMINI_API_KEY.value} not in os.environ")

        # Lazy session (create inside async call)
        self._session: Optional[aiohttp.ClientSession] = None

        self._api_key = api_key
        self._safety_settings = safety_settings or []
        # store a base url that follows earlier code's conventions (keeps trailing colon)
        # _model_name is provided by BaseLLMProvider
        self._base_url = GEMINI_API_BASE_URL + self._model_name + ":"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazily create and return an aiohttp.ClientSession (async-safe)."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers={"Content-Type": "application/json"})
        return self._session

    @classmethod
    def get_supported_models(cls) -> Union[list[str], str]:
        return ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"]

    def _ensure_url(self, url: Optional[str]) -> str:
        """Return a full endpoint URL for generateContent, building it if needed."""
        if isinstance(url, str) and url:
            return url

        base = getattr(self, "_base_url", None) or ""
        if not isinstance(base, str) or not base:
            # fallback to constant + model name
            base = GEMINI_API_BASE_URL + (getattr(self, "_model_name", "") or "") + ":"

        # normalize
        if base.endswith(":") or base.endswith("/"):
            candidate = base + "generateContent"
        else:
            candidate = base + ":generateContent"
        return candidate

    @backoff.on_exception(backoff.expo, BaseLLMProviderRateLimitException, max_value=10)
    @api_endpoint("generateContent")
    async def generate(
        self, prompt: str, url: Optional[str] = None, system_prompt: Optional[str] = None, **extra: Any
    ) -> Optional[BaseLLMProviderResponse]:
        """
        High-level generate API: accepts prompt text. `url` is optional; if not provided we'll build one.
        """
        url_final = self._ensure_url(url)

        messages = [BaseLLMMessage(role=LLMRole.USER, content=prompt)]
        if system_prompt is not None:
            messages.insert(0, BaseLLMMessage(role=LLMRole.SYSTEM, content=system_prompt))

        # forward to chat() which does the HTTP call
        return await self.chat(messages, url=url_final, system_prompt=system_prompt, **extra)  # type: ignore

    def sync_generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        raise NotImplementedError

    @api_endpoint("generateContent")
    async def chat(
        self, messages: list[BaseLLMMessage], url: Optional[str] = None, system_prompt: Optional[str] = None, **extra: Any
    ) -> BaseLLMProviderResponse:
        """
        Make the HTTP POST to Gemini using the session created in an async context.
        `url` is optional; if missing we compute a default from the provider model/base.
        """
        try:
            if system_prompt is not None and not any(m.role == LLMRole.SYSTEM for m in messages):
                messages.insert(0, BaseLLMMessage(role=LLMRole.SYSTEM, content=system_prompt))

            request = GenerateContentRequest.from_messages(messages, **extra)
            if self._safety_settings:
                request.safety_settings = self._safety_settings

            url_final = self._ensure_url(url)
            session = await self._get_session()
            async with session.post(url_final, data=request.model_dump_json(by_alias=True), params={"key": self._api_key}) as r:
                gemini_response = await r.json()
                if r.status != 200:
                    # include returned payload for diagnostics
                    raise GeminiProviderException(f"Error generating text: status={r.status} payload={gemini_response}")

                response: GenerateContentResponse = GenerateContentResponse.model_validate(gemini_response)
                return BaseLLMProviderResponse(response=response.get_content() or str())
        except (GeminiProviderException, BaseLLMProviderRateLimitException) as e:
            raise e
        except Exception as e:
            logger.exception("Error generating text from Gemini: %s", e)
            raise GeminiProviderException("Cant generate text") from e

    def sync_chat(self, messages: list[BaseLLMMessage], **extra: Any) -> Optional[BaseLLMProviderResponse]:
        raise NotImplementedError

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

import abc
import logging
from typing import Any, Optional, Type, TypeVar, Union, List

from pydantic import BaseModel

from fuzzyai.llm.chain import FuzzChain, FuzzNode
from fuzzyai.llm.models import BaseLLMProviderResponse
from fuzzyai.llm.providers.enums import LLMProvider
from fuzzyai.utils.flavor_manager import TypedFlavorManager

T = TypeVar('T')
KeyT = TypeVar('KeyT')
ValT = TypeVar('ValT')

logger = logging.getLogger(__name__)


class BaseLLMProviderRateLimitException(Exception):
    pass


class BaseLLMProviderException(Exception):
    pass


class BaseLLMMessage(BaseModel):
    role: str
    content: str


class BaseLLMProvider(abc.ABC):
    """
    Base class for Language Model Providers.

    Args:
        provider (LLMProvider): provider enum (e.g. LLMProvider.OLLAMA)
        model (str): provider-specific model identifier (e.g. 'qwen2.5' or local path)
        **extra (Any): Additional args passed to specific providers.
    """

    def __init__(self, provider: LLMProvider, model: str, **extra: Any) -> None:
        super().__init__()
        # qualified name: provider.value/model
        self._qualified_model_name = f"{provider.value}/{model}"
        self._model_name = model
        self._history: list[BaseLLMProviderResponse] = []
        self._provider = provider
        self._meta: dict[str, Any] = {}
        logger.debug("Initialized provider %s for model %s", provider.value, model)

    # --------- abstract runtime methods ----------
    @abc.abstractmethod
    async def generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        ...

    @abc.abstractmethod
    async def chat(self, messages: list[BaseLLMMessage], **extra: Any) -> Optional[BaseLLMProviderResponse]:
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        ...

    @abc.abstractmethod
    def sync_generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        ...

    @abc.abstractmethod
    def sync_chat(self, messages: list[BaseLLMMessage], **extra: Any) -> Optional[BaseLLMProviderResponse]:
        ...

    # --------- discovery API (should be implemented by subclasses) ----------
    @classmethod
    @abc.abstractmethod
    def get_supported_models(cls) -> Union[list[str], str]:
        """
        Return supported models for this provider.

        This is used by the UI to populate model lists. Subclasses must override.
        Return either a list of model identifiers or a single string describing how to use this provider.
        """
        ...

    # --------- default helpers (non-breaking additions) ----------
    def add_to_history(self, responses: list[BaseLLMProviderResponse]) -> None:
        self._history.extend(responses)

    def clear_history(self) -> None:
        self._history.clear()

    def get_history(self) -> list[BaseLLMProviderResponse]:
        return self._history

    @property
    def qualified_model_name(self) -> str:
        return self._qualified_model_name

    @property
    def history(self) -> list[BaseLLMProviderResponse]:
        return self._history

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    @property
    def meta(self) -> dict:
        return self._meta

    def is_model_valid(self, model: str) -> bool:
        """
        Optional runtime sanity-check for a model identifier.
        Subclasses can override for provider-specific validation (local path exists, REST file exists, etc).
        Default: True (non-restrictive).
        """
        return True

    def __or__(self, other: Union[FuzzNode, FuzzChain, str]) -> FuzzChain:
        if other.__class__ == FuzzChain:
            return FuzzChain([FuzzNode(self, "{input}"), *other._nodes])  # type: ignore
        elif other.__class__ == FuzzNode:
            return FuzzChain([FuzzNode(self, "{input}"), other])  # type: ignore
        elif other.__class__ == str:
            return FuzzChain([FuzzNode(self, other)])  # type: ignore

        raise ValueError(f"Invalid type for other: {other.__class__}")

    def __repr__(self) -> str:
        return f"<LLMProvider provider={self._provider.value} model={self._model_name}>"

    def __str__(self) -> str:
        return f"{self._provider.value}/{self._model_name}"


class ProviderFlavorManager(TypedFlavorManager[KeyT, ValT]):
    """
    Flavor manager used to register provider classes and safely query supported models.

    New helper: get_supported_models_safe(provider_enum) -> list[str]
    This calls the provider class's get_supported_models() with safety: exceptions are logged and empty list returned.
    """

    def __init__(self) -> None:
        super().__init__()

    def kwargs_type_parameter_name(self) -> str:
        return "provider"

    def get_registered_providers(self) -> List[str]:
        """
        Return a list of keys (registered provider names) currently present in the manager.
        Useful for UI display and diagnostics.
        """
        return list(self._map.keys())

    def get_supported_models_safe(self, provider_key: Union[str, LLMProvider]) -> Union[List[str], str]:
        """
        Safely call the provider class's get_supported_models() and return results.
        On any exception, logs the error and returns an empty list (UI-friendly).
        Accepts either the enum or the string key.
        """
        key = provider_key.value if isinstance(provider_key, LLMProvider) else str(provider_key)
        provider_cls = None
        try:
            provider_cls = self._map.get(key)
            if provider_cls is None:
                # try case-insensitive match
                for k, v in self._map.items():
                    if k.lower() == key.lower():
                        provider_cls = v
                        break
            if provider_cls is None:
                logger.debug("ProviderFlavorManager: provider %s not registered", key)
                return []
            # call the class method if available
            if hasattr(provider_cls, "get_supported_models"):
                result = provider_cls.get_supported_models()
                # normalize result to list if string returned
                if isinstance(result, str):
                    return [result]
                return result or []
            return []
        except Exception as e:
            logger.exception("Error retrieving supported models for provider %s: %s", key, e)
            return []


llm_provider_fm: ProviderFlavorManager[str, Type[BaseLLMProvider]] = ProviderFlavorManager()

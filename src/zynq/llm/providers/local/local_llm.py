import logging
from typing import Any, Optional, Union

# Lightweight imports only at module import time
from zynq.handlers.text_generation.llm_text_generator import LLMTextGenerationHandler
from zynq.handlers.tokenizers.handler import TokensHandler  # type: ignore
from zynq.llm.models import BaseLLMProviderResponse
from zynq.llm.providers.base import BaseLLMMessage, BaseLLMProvider, BaseLLMProviderException, llm_provider_fm
from zynq.llm.providers.enums import LLMProvider
from zynq.llm.providers.local.models import LocalGenerateOptions

logger = logging.getLogger(__name__)


class LocalProviderException(BaseLLMProviderException):
    pass


# Helper: lazy-import heavy dependencies (transformers, torch)
def _lazy_heavy_imports():
    """
    Import heavy ML libs only when needed. Raises a helpful error if not available.
    Returns tuple: (AutoModelForCausalLM, AutoTokenizer, torch)
    """
    try:
        import torch as _torch
    except Exception as e:
        raise LocalProviderException(
            "The 'torch' package is required for LocalProvider but is not importable. "
            "Install PyTorch for your platform (see https://pytorch.org/) and try again. "
            f"Original error: {e}"
        ) from e

    try:
        # Import from transformers lazily
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM, AutoTokenizer as _AutoTokenizer
    except Exception as e:
        raise LocalProviderException(
            "The 'transformers' package is required for LocalProvider but is not importable. "
            "Install it in your environment (e.g. `pip install transformers`) and ensure any binary deps "
            "(scipy/sklearn) are present. Original error: "
            f"{e}"
        ) from e

    return _AutoModelForCausalLM, _AutoTokenizer, _torch


@llm_provider_fm.flavor(LLMProvider.LOCAL)
class LocalProvider(BaseLLMProvider):
    """
    Local provider that loads transformer models lazily.

    Constructor only stores configuration; heavy model/tokenizer load happens
    in _ensure_model_loaded() when generate/chat is first called.
    """
    def __init__(self, model: str, tokenizer_path: Optional[str] = None, device: str = "cuda:0", **kwargs: Any) -> None:
        # Keep light-weight initialization only
        super().__init__(model=model, **kwargs)
        self._device = device
        # keep provided args (don't delete provider blindly if not present)
        if "provider" in kwargs:
            try:
                del kwargs["provider"]
            except Exception:
                pass

        # store config to use later during actual model load
        self._model_name = model
        self._tokenizer_path_config = model if tokenizer_path is None else tokenizer_path
        self._model_kwargs = kwargs

        # placeholders for heavy objects
        self._auto_model = None
        self._tokenizer = None
        self._tokens_handler: Optional[TokensHandler] = None
        self._text_gen_handler: Optional[LLMTextGenerationHandler] = None

    @classmethod
    def get_supported_models(cls) -> Union[list[str], str]:
        return "<Full path to model>"

    def _ensure_model_loaded(self) -> None:
        """
        Load tokenizer + model only when first needed.
        Throws LocalProviderException with actionable advice on failures.
        """
        if self._auto_model is not None and self._tokenizer is not None:
            return

        AutoModelForCausalLM, AutoTokenizer, torch = _lazy_heavy_imports()

        try:
            # load tokenizer then model (loading can be slow and memory-heavy)
            logger.info(f"Loading tokenizer from '{self._tokenizer_path_config}' (this may take a while)...")
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path_config, trust_remote_code=True, use_fast=True)
        except Exception as e:
            raise LocalProviderException(
                f"Failed to load tokenizer from '{self._tokenizer_path_config}'. "
                "Make sure the path/model name is correct and required files are available. "
                f"Original error: {e}"
            ) from e

        try:
            logger.info(f"Loading model '{self._model_name}' to device {self._device} (this may take a while)...")
            # Allow model kwargs passed via CLI or config
            model_kwargs = dict(self._model_kwargs or {})
            # If user didn't pass dtype, attempt to use float16 only if device is CUDA
            if "torch_dtype" not in model_kwargs:
                if "cuda" in str(self._device).lower():
                    # avoid importing torch at module level — we imported it here
                    model_kwargs["torch_dtype"] = getattr(torch, "float16", None)
            # trust_remote_code kept as default True in original
            self._auto_model = AutoModelForCausalLM.from_pretrained(self._model_name, trust_remote_code=True, **model_kwargs)
            # move to device if applicable
            try:
                self._auto_model = self._auto_model.to(self._device).eval()
            except Exception:
                # if device move fails, keep model on default device and warn
                logger.warning("Could not move model to requested device; continuing with default device.")
        except Exception as e:
            raise LocalProviderException(
                f"Failed to load model '{self._model_name}'. Ensure model path/name is correct and dependencies (torch/transformers) are installed. Original error: {e}"
            ) from e

        # Create tokens handler and text generation handler now that tokenizer & model are present
        try:
            self._tokens_handler = TokensHandler(tokenizer=self._tokenizer)
            self._text_gen_handler = LLMTextGenerationHandler(self._auto_model, self._tokenizer)
        except Exception as e:
            raise LocalProviderException(f"Failed to initialize tokenizer/text generation helpers: {e}") from e

        logger.info("LocalProvider loaded successfully.")

    def sync_generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        try:
            """Generate text synchronously using the model."""
            self._ensure_model_loaded()
            if self._text_gen_handler is None:
                raise LocalProviderException("Text generation handler not initialized")
            response = self._text_gen_handler.generate_text(prompt, **extra)
            return BaseLLMProviderResponse(response=response)
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise LocalProviderException("Cant generate text") from e

    async def generate(self, prompt: str, **extra: Any) -> Optional[BaseLLMProviderResponse]:
        try:
            self._ensure_model_loaded()
            if self._tokenizer is None or self._auto_model is None:
                raise LocalProviderException("Model or tokenizer not initialized")
            options = LocalGenerateOptions.model_validate(extra)
            inputs = self._tokenizer.encode(prompt, return_tensors="pt")
            # make sure tensors use same device as model if model on cuda
            try:
                output = self._auto_model.generate(inputs.to(self._device), num_return_sequences=1, **options.model_dump())
            except Exception:
                # fallback: try CPU if device fail
                output = self._auto_model.generate(inputs, num_return_sequences=1, **options.model_dump())
            text_output = self._tokenizer.decode(output[0], skip_special_tokens=True)
            # return the continuation only
            return BaseLLMProviderResponse(response=text_output[len(prompt) :])
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise LocalProviderException("Cant generate text") from e

    async def chat(self, messages: list[BaseLLMMessage], **extra: Any) -> BaseLLMProviderResponse | None:
        try:
            self._ensure_model_loaded()
            if self._tokenizer is None or self._auto_model is None:
                raise LocalProviderException("Model or tokenizer not initialized")
            options = LocalGenerateOptions.model_validate(extra)
            full_prompt = self._tokenizer.apply_chat_template([m.model_dump() for m in messages], tokenize=False)
            inputs = self._tokenizer.encode(full_prompt, return_tensors="pt")
            try:
                output = self._auto_model.generate(inputs.to(self._device), num_return_sequences=1, **options.model_dump())
            except Exception:
                output = self._auto_model.generate(inputs, num_return_sequences=1, **options.model_dump())
            text_output = self._tokenizer.decode(output[0], skip_special_tokens=True)
            return BaseLLMProviderResponse(response=text_output[len(full_prompt) :])
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise LocalProviderException("Cant generate text") from e

    def sync_chat(self, messages: list[BaseLLMMessage], **extra: Any) -> Optional[BaseLLMProviderResponse]:
        raise NotImplementedError

    async def close(self) -> None:
        # Optionally release model memory if needed
        try:
            if self._auto_model is not None:
                # If the model supports .cpu() / .to, try to move to cpu and delete
                try:
                    self._auto_model.to("cpu")
                except Exception:
                    pass
            # delete references
            self._auto_model = None
            self._tokenizer = None
            self._tokens_handler = None
            self._text_gen_handler = None
        except Exception:
            pass

"""
LLM text generation helper with lazy imports so importing this module is cheap.
"""
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def _lazy_transformers_imports():
    try:
        import torch as _torch
    except Exception as e:
        raise RuntimeError("Missing dependency 'torch'. Install PyTorch to use local models.") from e

    try:
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM, AutoTokenizer as _AutoTokenizer
    except Exception as e:
        raise RuntimeError("Missing dependency 'transformers'. Install transformers to use local models.") from e

    return _AutoModelForCausalLM, _AutoTokenizer, _torch


class LLMTextGenerationHandler:
    """
    Thin wrapper around model + tokenizer. This module doesn't import transformers at top-level.
    The caller should pass loaded model/tokenizer when possible, or call build_from_names.
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = getattr(model, "device", None)

    @classmethod
    def build_from_names(cls, model_name: str, tokenizer_name: Optional[str] = None, device: Optional[str] = None, model_kwargs: Optional[dict] = None):
        AutoModelForCausalLM, AutoTokenizer, torch = _lazy_transformers_imports()
        tokenizer_name = tokenizer_name or model_name
        model_kwargs = model_kwargs or {}
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, use_fast=True)
        auto_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **model_kwargs)
        if device:
            try:
                auto_model = auto_model.to(device).eval()
            except Exception:
                logger.warning("Could not move model to requested device; continuing with default device.")
        return cls(model=auto_model, tokenizer=tokenizer)

    def generate_text(self, prompt: str, max_length: int = 30, temperature: float = 1e-6, include_prompt: bool = False) -> Any:
        if temperature <= 0:
            raise ValueError("Temperature should be greater than 0")
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model/tokenizer not loaded. Use build_from_names(...) or pass loaded model/tokenizer to constructor.")
        # Encode input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        self.model.eval()
        with self._no_grad():
            outputs = self.model.generate(inputs, max_length=max_length, pad_token_id=getattr(self.tokenizer, "eos_token_id", None), temperature=temperature)
            ids_to_decode = outputs[0] if include_prompt else outputs[0, inputs.shape[-1] :]
            generated_text = self.tokenizer.decode(ids_to_decode, skip_special_tokens=True)
        return generated_text

    # small helper to allow mocking if needed
    def _no_grad(self):
        # Import torch lazily if needed
        try:
            import torch
            return torch.no_grad()
        except Exception as e:
            raise RuntimeError("Missing dependency 'torch'. Install PyTorch to use local models.") from e

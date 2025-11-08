"""
Tokenizer helpers with lazy imports.
Works with HuggingFace tokenizers or tiktoken encodings.
"""
from typing import Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


def _lazy_auto_tokenizer():
    try:
        from transformers import AutoTokenizer as _AutoTokenizer
    except Exception as e:
        raise RuntimeError("Missing dependency 'transformers'. Install transformers to use tokenizer helpers.") from e
    return _AutoTokenizer


class TokensHandler:
    def __init__(self, tokenizer: Optional[Any] = None, tokenizer_name: Optional[str] = None):
        """
        tokenizer can be:
          - a HuggingFace tokenizer instance
          - a tiktoken.Encoding instance (if you use tiktoken)
          - None (in which case tokenizer_name will be used to lazy-load)
        """
        self._tokenizer = tokenizer
        self._tokenizer_name = tokenizer_name

    def _ensure_tokenizer(self):
        if self._tokenizer is not None:
            return
        AutoTokenizer = _lazy_auto_tokenizer()
        if not self._tokenizer_name:
            raise RuntimeError("Tokenizer name not provided for lazy loading")
        self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name, trust_remote_code=True, use_fast=True)

    def encode(self, text: str, **kwargs):
        self._ensure_tokenizer()
        return self._tokenizer.encode(text, **kwargs)

    def decode(self, tokens, **kwargs):
        self._ensure_tokenizer()
        return self._tokenizer.decode(tokens, **kwargs)

    def batch_decode(self, sequences, **kwargs):
        self._ensure_tokenizer()
        # keep compatibility with HF tokenizer.batch_decode if available
        if hasattr(self._tokenizer, "batch_decode"):
            return self._tokenizer.batch_decode(sequences, **kwargs)
        # otherwise decode one by one
        return [self._tokenizer.decode(seq, **kwargs) for seq in sequences]

    def apply_chat_template(self, messages: list[dict], tokenize: bool = True) -> str:
        self._ensure_tokenizer()
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(messages, tokenize=tokenize)
            except Exception:
                logger.debug("tokenizer.apply_chat_template failed, falling back to fallback join")
        return "\n".join((m.get("role", "") + ": " + m.get("content", "")) for m in messages)

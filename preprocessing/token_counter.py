"""
Healthcare Knowledge Navigator — Token Counter.

Centralized token-counting abstraction using tiktoken for fast,
accurate token estimation during chunk sizing.

Uses the cl100k_base encoding (GPT-4 compatible) as a general-purpose
approximation. The actual downstream embedding model may use a
different tokenizer, but for *sizing* chunks this is more than adequate.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken


class TokenCounter:
    """Wrapper around tiktoken for consistent token counting.

    Attributes:
        encoding: The tiktoken encoding instance.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        """Initialize the token counter.

        Args:
            encoding_name: The tiktoken encoding to use.
                           Defaults to cl100k_base (GPT-4).
        """
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Args:
            text: The input text to tokenize.

        Returns:
            The number of tokens.
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to a maximum number of tokens.

        Useful for hard-splitting oversized sentences.

        Args:
            text: The input text.
            max_tokens: Maximum number of tokens to keep.

        Returns:
            The truncated text decoded back to a string.
        """
        if not text:
            return ""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[:max_tokens])


@lru_cache(maxsize=1)
def get_token_counter() -> TokenCounter:
    """Return a cached singleton TokenCounter instance.

    Returns:
        TokenCounter: The shared token counter.
    """
    return TokenCounter()

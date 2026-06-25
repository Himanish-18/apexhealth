"""
Tests for the token counter (preprocessing.token_counter).
"""

from preprocessing.token_counter import TokenCounter, get_token_counter


def test_count_tokens_positive_for_nonempty():
    """Non-empty text should return a positive token count."""
    counter = TokenCounter()
    result = counter.count_tokens("Hello, world!")
    assert result > 0


def test_count_tokens_zero_for_empty():
    """Empty string should return 0 tokens."""
    counter = TokenCounter()
    assert counter.count_tokens("") == 0


def test_count_tokens_none_returns_zero():
    """None-ish empty input should return 0."""
    counter = TokenCounter()
    assert counter.count_tokens("") == 0


def test_count_tokens_consistent():
    """Same input should always produce the same token count."""
    counter = TokenCounter()
    text = "The patient was administered 500mg of acetaminophen."
    count_a = counter.count_tokens(text)
    count_b = counter.count_tokens(text)
    assert count_a == count_b


def test_truncate_to_tokens():
    """Truncation should produce text with at most max_tokens tokens."""
    counter = TokenCounter()
    long_text = "word " * 500  # ~500 tokens
    truncated = counter.truncate_to_tokens(long_text, 100)
    assert counter.count_tokens(truncated) <= 100


def test_truncate_short_text_unchanged():
    """Text shorter than max_tokens should be returned unchanged."""
    counter = TokenCounter()
    short_text = "Hello"
    result = counter.truncate_to_tokens(short_text, 100)
    assert result == short_text


def test_get_token_counter_singleton():
    """get_token_counter should return the same instance."""
    a = get_token_counter()
    b = get_token_counter()
    assert a is b

# tests/test_sentiment.py
import pytest
from src.sentiment import analyse_sentiment


def test_positive_review():
    """Clearly positive review should return positive."""
    result = analyse_sentiment(
        "This medication completely changed my life for the better!"
    )
    assert result["label"] == "positive"
    assert result["score"] > 0.5


def test_negative_review():
    """Clearly negative review should return negative."""
    result = analyse_sentiment(
        "Terrible side effects, I stopped taking it immediately."
    )
    assert result["label"] == "negative"
    assert result["score"] > 0.5


def test_output_structure():
    """Output must always have label and score."""
    result = analyse_sentiment("The medication was okay.")
    assert "label" in result
    assert "score" in result


def test_label_is_valid():
    """Label must be one of three valid values."""
    result = analyse_sentiment("I feel somewhat better.")
    assert result["label"] in ["positive", "negative", "neutral"]


def test_score_range():
    """Confidence score must be between 0 and 1."""
    result = analyse_sentiment("The drug helped with my pain.")
    assert 0.0 <= result["score"] <= 1.0


def test_empty_input():
    """Empty input should return unknown gracefully."""
    result = analyse_sentiment("")
    assert result["label"] == "unknown"
    assert result["score"] == 0.0


def test_long_input():
    """Long text should be handled without crashing."""
    long_text = "This medication helped me. " * 100
    result = analyse_sentiment(long_text)
    assert result["label"] in ["positive", "negative", "neutral"]
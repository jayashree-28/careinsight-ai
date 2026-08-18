# tests/test_topics.py
import pytest
from src.topics import load_reviews
from src.pipeline import extract_top_keywords


def test_load_reviews_returns_list():
    """load_reviews should return a list."""
    reviews = load_reviews("data/drugsComTrain_raw.csv", n_samples=10)
    assert isinstance(reviews, list)


def test_load_reviews_count():
    """load_reviews should return correct number of reviews."""
    reviews = load_reviews("data/drugsComTrain_raw.csv", n_samples=10)
    assert len(reviews) == 10


def test_reviews_are_strings():
    """All reviews should be strings."""
    reviews = load_reviews("data/drugsComTrain_raw.csv", n_samples=5)
    for review in reviews:
        assert isinstance(review, str)


def test_reviews_not_empty():
    """No review should be empty after cleaning."""
    reviews = load_reviews("data/drugsComTrain_raw.csv", n_samples=5)
    for review in reviews:
        assert len(review.strip()) > 0


def test_keyword_extraction_returns_list():
    """extract_top_keywords should return a list."""
    reviews = [
        "The medication helped with my anxiety and depression.",
        "Side effects were nausea and headache.",
        "Doctor prescribed this for pain management."
    ]
    keywords = extract_top_keywords(reviews)
    assert isinstance(keywords, list)


def test_keyword_extraction_count():
    """Should return up to 10 keywords."""
    reviews = [
        "The medication helped with my anxiety and depression.",
        "Side effects were nausea and headache.",
        "Doctor prescribed this for pain management.",
        "I have been taking this for 3 months now.",
        "The side effects subsided after a week."
    ]
    keywords = extract_top_keywords(reviews)
    assert len(keywords) <= 10


def test_keywords_are_strings():
    """All keywords should be strings."""
    reviews = ["Pain medication helped reduce my symptoms significantly."]
    keywords = extract_top_keywords(reviews)
    for kw in keywords:
        assert isinstance(kw, str)
# tests/test_pipeline.py
import pytest
from src.sentiment import analyse_sentiment
from src.summariser import clean_text, load_summariser, summarise_general
from src.pipeline import extract_top_keywords, run_pipeline


def test_clean_text_removes_html():
    """clean_text should remove HTML entities."""
    dirty = "I&#039;m feeling better &amp; less anxious"
    clean = clean_text(dirty)
    assert "&#039;" not in clean
    assert "&amp;" not in clean
    assert "I'm" in clean


def test_clean_text_strips_whitespace():
    """clean_text should strip leading/trailing whitespace."""
    text = "  This medication helped me.  "
    result = clean_text(text)
    assert result == result.strip()


def test_pipeline_output_structure():
    """Pipeline output must have required keys."""
    tokenizer, model = load_summariser()
    reviews = [
        "This medication helped with my depression significantly.",
        "Side effects were minimal and manageable.",
        "I feel much better after 3 months of treatment."
    ]
    result = run_pipeline(reviews, tokenizer, model)
    assert "sentiment" in result
    assert "top_topics" in result
    assert "general_summary" in result
    assert "medgemma_summary" in result


def test_pipeline_sentiment_structure():
    """Sentiment in pipeline output must have label and confidence."""
    tokenizer, model = load_summariser()
    reviews = ["This drug completely eliminated my anxiety symptoms."]
    result = run_pipeline(reviews, tokenizer, model)
    assert "label" in result["sentiment"]
    assert "confidence" in result["sentiment"]


def test_pipeline_topics_is_list():
    """Topics in pipeline output must be a list."""
    tokenizer, model = load_summariser()
    reviews = [
        "Pain medication helped reduce my chronic pain.",
        "The doctor prescribed morphine for back pain.",
        "Side effects included nausea and dizziness."
    ]
    result = run_pipeline(reviews, tokenizer, model)
    assert isinstance(result["top_topics"], list)


def test_pipeline_summary_not_empty():
    """General summary must not be empty."""
    tokenizer, model = load_summariser()
    reviews = [
        "This medication worked well for my condition.",
        "I noticed improvement within the first week.",
        "My doctor was pleased with the results."
    ]
    result = run_pipeline(reviews, tokenizer, model)
    assert len(result["general_summary"]) > 0
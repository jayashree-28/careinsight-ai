# tests/test_data_quality.py
import pytest
import pandas as pd


@pytest.fixture
def df():
    """Load dataset once for all tests."""
    return pd.read_csv(
        "data/drugsComTrain_raw.csv",
        sep=",",
        on_bad_lines="skip",
        encoding="utf-8"
    )


def test_required_columns_exist(df):
    """Dataset must have all required columns."""
    required = ["drugName", "condition", "review", "rating"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_no_null_reviews(df):
    """Less than 5% of reviews should be null."""
    null_pct = df["review"].isnull().mean()
    assert null_pct < 0.05, f"Too many null reviews: {null_pct:.1%}"


def test_rating_range(df):
    """All ratings must be between 1 and 10."""
    valid = df["rating"].dropna()
    assert valid.min() >= 1, "Rating below 1 found"
    assert valid.max() <= 10, "Rating above 10 found"


def test_review_minimum_length(df):
    """Reviews should be at least 10 characters."""
    short = df["review"].dropna().str.len() < 10
    assert short.sum() < 200, f"Too many very short reviews: {short.sum()}"


def test_dataset_size(df):
    """Dataset must have at least 1000 reviews."""
    assert len(df) >= 1000, f"Dataset too small: {len(df)} rows"


def test_drug_name_not_empty(df):
    """Drug names should mostly be populated."""
    null_pct = df["drugName"].isnull().mean()
    assert null_pct < 0.10, f"Too many null drug names: {null_pct:.1%}"


def test_condition_column_exists(df):
    """Condition column must have valid entries."""
    non_null = df["condition"].dropna()
    assert len(non_null) > 0, "No conditions found"


def test_sentiment_output_schema():
    """Sentiment output must match expected schema."""
    from src.sentiment import analyse_sentiment
    result = analyse_sentiment("This medication helped me feel better.")
    assert isinstance(result, dict)
    assert isinstance(result["label"], str)
    assert isinstance(result["score"], float)
    assert result["label"] in ["positive", "negative", "neutral", "unknown"]
    assert 0.0 <= result["score"] <= 1.0


def test_pipeline_output_schema():
    """Pipeline output must match expected schema."""
    from src.summariser import load_summariser
    from src.pipeline import run_pipeline
    tokenizer, model = load_summariser()
    reviews = ["This medication helped reduce my pain significantly."]
    result = run_pipeline(reviews, tokenizer, model)
    assert isinstance(result["sentiment"], dict)
    assert isinstance(result["top_topics"], list)
    assert isinstance(result["general_summary"], str)
    assert len(result["general_summary"]) > 0
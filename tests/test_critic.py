
from types import SimpleNamespace

from careinsight.critic_tools import (
    validate_medication_report,
    validate_report_evidence,
)


def make_valid_report():
    """Build a consistent evidence package for validator tests."""
    return {
        "medication": "TestMedication",
        "report_status": "evidence_collected",
        "retrieval_analysis": {
            "total_reviews": 100,
            "rating_distribution": {"5": 100},
            "average_rating": 5.0,
            "supporting_reviews": ["Example retrieval review"],
        },
        "sentiment_analysis": {
            "total_reviews_analyzed": 100,
            "sentiment_distribution": {
                "positive": {"count": 50, "percentage": 50.0},
                "neutral": {"count": 25, "percentage": 25.0},
                "negative": {"count": 25, "percentage": 25.0},
            },
            "supporting_reviews": ["Example sentiment review"],
        },
        "topic_analysis": {
            "reviews_analyzed": 100,
            "number_of_topics": 2,
            "outlier_reviews": 0,
            "topics": [
                {"review_count": 60},
                {"review_count": 40},
            ],
        },
    }


def test_valid_report_is_approved():
    result = validate_medication_report(make_valid_report())

    assert result["status"] == "APPROVED"
    assert result["checks_passed"] > 0
    assert result["checks_total"] >= result["checks_passed"]
    assert result["errors"] == []


def test_missing_evidence_requires_revision():
    report = make_valid_report()
    report["report_status"] = "failed"

    result = validate_medication_report(report)

    assert result["status"] == "REVISION_REQUIRED"
    assert result["errors"]


def test_inconsistent_rating_distribution_requires_revision():
    report = make_valid_report()
    report["retrieval_analysis"]["rating_distribution"] = {"5": 90}

    result = validate_medication_report(report)

    assert result["status"] == "REVISION_REQUIRED"
    assert any(
        "Rating distribution" in error
        for error in result["errors"]
    )


def test_inconsistent_sentiment_counts_require_revision():
    report = make_valid_report()
    report["sentiment_analysis"]["sentiment_distribution"][
        "positive"
    ]["count"] = 40

    result = validate_medication_report(report)

    assert result["status"] == "REVISION_REQUIRED"
    assert any(
        "Sentiment count reconciliation" in error
        for error in result["errors"]
    )


def test_skipped_topic_modeling_generates_warning():
    report = make_valid_report()
    report["retrieval_analysis"]["total_reviews"] = 1
    report["retrieval_analysis"]["rating_distribution"] = {"10": 1}
    report["retrieval_analysis"]["average_rating"] = 10.0

    report["sentiment_analysis"]["total_reviews_analyzed"] = 1
    report["sentiment_analysis"]["sentiment_distribution"] = {
        "positive": {"count": 1, "percentage": 100.0},
        "neutral": {"count": 0, "percentage": 0.0},
        "negative": {"count": 0, "percentage": 0.0},
    }

    report["topic_analysis"] = {
        "reviews_analyzed": 1,
        "number_of_topics": 0,
        "outlier_reviews": None,
        "topics": [],
    }

    result = validate_medication_report(report)

    assert result["status"] == "APPROVED"
    assert any(
        "Topic modeling skipped" in warning
        for warning in result["warnings"]
    )


def test_topic_count_mismatch_requires_revision():
    report = make_valid_report()
    report["topic_analysis"]["topics"][0]["review_count"] = 50

    result = validate_medication_report(report)

    assert result["status"] == "REVISION_REQUIRED"
    assert any(
        "Topic count reconciliation" in error
        for error in result["errors"]
    )


def test_validate_report_evidence_reads_session_state():
    report = make_valid_report()
    tool_context = SimpleNamespace(
        state={"careinsight_evidence": report}
    )

    result = validate_report_evidence(tool_context)

    assert result["status"] == "APPROVED"


def test_validate_report_evidence_handles_missing_state():
    tool_context = SimpleNamespace(state={})

    result = validate_report_evidence(tool_context)

    assert result["status"] == "REVISION_REQUIRED"
    assert result["checks_total"] == 0
    assert result["errors"]


from google.adk.tools import ToolContext

from careinsight.retrieval import search_medication_reviews
from careinsight.sentiment_tools import analyze_medication_sentiment
from careinsight.topic_tools import analyze_medication_topics


def generate_medication_report(
    medication_name: str,
    tool_context: ToolContext,
) -> dict:
    """
    Collect retrieval, sentiment, and topic analysis
    into one shared evidence package.

    Store the package in ADK session state so the
    Critic Agent can validate the same evidence.
    """

    if not medication_name or not medication_name.strip():
        return {"error": "Medication name is required."}

    medication_name = medication_name.strip()

    try:
        retrieval_results = search_medication_reviews(
            medication_name=medication_name,
            max_examples=5,
        )

        sentiment_results = analyze_medication_sentiment(
            medication_name=medication_name,
            batch_size=16,
        )

        topic_results = analyze_medication_topics(
            medication_name=medication_name,
        )

        report_data = {
            "medication": medication_name,
            "retrieval_analysis": retrieval_results,
            "sentiment_analysis": sentiment_results,
            "topic_analysis": topic_results,
            "report_status": "evidence_collected",
            "limitations": [
                "Sentiment reflects language in patient reviews, "
                "not verified clinical outcomes.",
                "Topics are exploratory patterns, "
                "not verified medical findings.",
                "Ratings should not be treated as sentiment labels.",
                "This report does not provide medical advice.",
            ],
        }

        # Save the evidence package for the Critic Agent.
        tool_context.state["careinsight_evidence"] = report_data

        return report_data

    except Exception as error:
        return {
            "medication": medication_name,
            "report_status": "failed",
            "error": str(error),
        }

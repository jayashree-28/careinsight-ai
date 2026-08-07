# src/pipeline.py
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from src.sentiment import analyse_sentiment
from src.topics import load_reviews, extract_topics, get_topic_summary
from src.summariser import clean_text, load_summariser, summarise_general


def build_pipeline(tokenizer, model):
    """
    Build LangChain pipeline that chains:
    sentiment → topics → summary
    """

    # Step 1 — Sentiment analysis
    sentiment_step = RunnableLambda(
        lambda x: {
            **x,
            "sentiment": analyse_sentiment(x["combined_text"])
        }
    )

    # Step 2 — Topic extraction
    topic_step = RunnableLambda(
        lambda x: {
            **x,
            "topics": extract_top_keywords(x["reviews"])
        }
    )

    # Step 3 — Summarisation
    summary_step = RunnableLambda(
        lambda x: {
            **x,
            "general_summary": summarise_general(
                x["reviews"], tokenizer, model
            )
        }
    )

    # Step 4 — Format final output
    output_step = RunnableLambda(format_output)

    # Chain all steps together
    pipeline = (
        sentiment_step
        | topic_step
        | summary_step
        | output_step
    )

    return pipeline


def extract_top_keywords(reviews: list) -> list:
    """
    Extract simple top keywords from reviews
    without full BERTopic (faster for pipeline).
    """
    from sklearn.feature_extraction.text import CountVectorizer
    
    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=10,
        ngram_range=(1, 2)
    )
    
    try:
        vectorizer.fit_transform(reviews)
        keywords = vectorizer.get_feature_names_out().tolist()
    except Exception:
        keywords = ["Unable to extract topics"]
    
    return keywords


def format_output(x: dict) -> dict:
    """
    Format the final pipeline output cleanly.
    """
    return {
        "sentiment": {
            "label": x["sentiment"]["label"].upper(),
            "confidence": f"{x['sentiment']['score'] * 100:.1f}%"
        },
        "top_topics": x["topics"],
        "general_summary": x["general_summary"],
        "medgemma_summary": "Coming in Phase 3..."
    }


def run_pipeline(reviews: list, tokenizer, model) -> dict:
    """
    Run the full CareInsight AI pipeline on a list of reviews.
    """
    # Clean reviews
    reviews = [clean_text(r) for r in reviews]
    
    # Combine for sentiment analysis
    combined_text = " ".join(reviews)[:512]
    
    # Build and run pipeline
    pipeline = build_pipeline(tokenizer, model)
    
    result = pipeline.invoke({
        "reviews": reviews,
        "combined_text": combined_text
    })
    
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("CareInsight AI — Full Pipeline Test")
    print("=" * 60)

    # Load summariser model
    print("\nLoading models...")
    tokenizer, model = load_summariser()

    # Load sample reviews
    df = pd.read_csv("data/drugsComTrain_raw.csv",
                     sep=",",
                     on_bad_lines="skip",
                     encoding="utf-8")

    # Get 10 Depression reviews
    sample_reviews = (df[df["condition"] == "Depression"]
                      ["review"]
                      .dropna()
                      .head(10)
                      .tolist())

    print(f"\n📋 Running full pipeline on 10 Depression reviews...")

    # Run full pipeline
    result = run_pipeline(sample_reviews, tokenizer, model)

    # Print results
    print("\n" + "=" * 60)
    print("🎯 CAREINSIGHT AI — FULL PIPELINE RESULTS")
    print("=" * 60)

    print(f"\n😊 SENTIMENT: {result['sentiment']['label']}")
    print(f"   Confidence: {result['sentiment']['confidence']}")

    print(f"\n🏷️  TOP TOPICS:")
    for topic in result["top_topics"]:
        print(f"   • {topic}")

    print(f"\n📄 GENERAL MODEL SUMMARY:")
    print(f"   {result['general_summary']}")

    print(f"\n🏥 MEDGEMMA SUMMARY:")
    print(f"   {result['medgemma_summary']}")

    print("\n✅ Full pipeline complete!")
# src/sentiment.py
from transformers import pipeline

# Load RoBERTa sentiment model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True,
    max_length=512
)

def analyse_sentiment(text: str) -> dict:
    """
    Analyse sentiment of a patient review.
    Returns label (positive/neutral/negative) and confidence score.
    """
    if not text or not isinstance(text, str):
        return {"label": "unknown", "score": 0.0}
    
    # Clean text
    text = text.strip().replace("\n", " ")
    
    # Run sentiment analysis
    result = sentiment_pipeline(text)[0]
    
    return {
        "label": result["label"].lower(),
        "score": round(result["score"], 4)
    }

if __name__ == "__main__":
    # Quick test with sample patient reviews
    import pandas as pd
    
    df = pd.read_csv("data/drugsComTrain_raw.csv",
                     sep=",",
                     on_bad_lines="skip",
                     encoding="utf-8")
    
    # Test on 5 sample reviews
    samples = df["review"].dropna().head(5).tolist()
    
    print("=" * 60)
    print("CareInsight AI — Sentiment Analysis Test")
    print("=" * 60)
    
    for i, review in enumerate(samples):
        result = analyse_sentiment(review)
        print(f"\nReview {i+1}: {review[:100]}...")
        print(f"Sentiment: {result['label'].upper()}")
        print(f"Confidence: {result['score']}")
        print("-" * 40)
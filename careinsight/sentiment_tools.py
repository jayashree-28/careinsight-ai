
from pathlib import Path

import pandas as pd

from src.sentiment import sentiment_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILES = [
    PROJECT_ROOT / "data" / "drugsComTrain_raw.csv",
    PROJECT_ROOT / "data" / "drugsComTest_raw.csv",
]


def analyze_medication_sentiment(
    medication_name: str,
    batch_size: int = 16,
) -> dict:
    """
    Analyze sentiment across patient reviews for one medication
    using the existing CardiffNLP RoBERTa model.
    """

    if not medication_name or not medication_name.strip():
        return {"error": "Please provide a medication name."}

    if not 1 <= batch_size <= 64:
        return {"error": "batch_size must be between 1 and 64."}

    frames = []

    for file_path in DATA_FILES:
        if not file_path.exists():
            return {"error": f"Dataset file not found: {file_path}"}

        df = pd.read_csv(file_path, on_bad_lines="skip")
        frames.append(df)

    reviews = pd.concat(frames, ignore_index=True)

    medication = medication_name.strip().casefold()

    matches = reviews[
        reviews["drugName"]
        .astype(str)
        .str.strip()
        .str.casefold()
        == medication
    ].copy()

    matches = matches.dropna(subset=["review"])

    if matches.empty:
        return {
            "medication": medication_name,
            "total_reviews": 0,
            "message": "No matching reviews were found.",
        }

    texts = (
        matches["review"]
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.strip()
        .tolist()
    )

    predictions = sentiment_pipeline(
        texts,
        batch_size=batch_size,
        truncation=True,
        max_length=512,
    )

    matches["sentiment"] = [
        result["label"].lower() for result in predictions
    ]

    matches["confidence"] = [
        round(float(result["score"]), 4) for result in predictions
    ]

    sentiment_counts = (
        matches["sentiment"]
        .value_counts()
        .reindex(["positive", "neutral", "negative"], fill_value=0)
    )

    total = len(matches)

    sentiment_summary = {}

    for label, count in sentiment_counts.items():
        sentiment_summary[label] = {
            "count": int(count),
            "percentage": round(float(count / total * 100), 2),
        }

    examples = []

    for _, row in matches.head(5).iterrows():
        examples.append({
            "review_id": int(row["uniqueID"]),
            "sentiment": row["sentiment"],
            "confidence": row["confidence"],
            "rating": int(row["rating"]),
            "review": str(row["review"])[:300],
        })

    return {
        "medication": medication_name,
        "total_reviews_analyzed": total,
        "sentiment_distribution": sentiment_summary,
        "supporting_reviews": examples,
        "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "note": (
            "Sentiment labels represent model predictions about review text. "
            "They do not establish clinical outcomes or confirm side effects."
        ),
    }

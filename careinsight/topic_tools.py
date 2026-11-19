
from pathlib import Path

import pandas as pd

from src.topics import extract_topics, get_topic_summary

import html

import json

import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILES = [
    PROJECT_ROOT / "data" / "drugsComTrain_raw.csv",
    PROJECT_ROOT / "data" / "drugsComTest_raw.csv",
]

def analyze_medication_topics(
    medication_name: str,
    max_reviews: int | None = None,
) -> dict:
    """
    Discover recurring themes in patient reviews for one medication
    using the existing BERTopic implementation.

    By default, analyze all available reviews.
    An optional max_reviews parameter can limit processing.
    """

    if not medication_name or not medication_name.strip():
        return {"error": "Please provide a medication name."}

    if max_reviews is not None and not 50 <= max_reviews <= 5000:
        return {"error": "max_reviews must be between 50 and 5000."}

    frames = []

    for file_path in DATA_FILES:
        if not file_path.exists():
            return {"error": f"Dataset file not found: {file_path}"}

        df = pd.read_csv(file_path, on_bad_lines="skip")
        frames.append(df)

    reviews_df = pd.concat(frames, ignore_index=True)

    medication = medication_name.strip().casefold()

    safe_name = re.sub(r"[^a-z0-9_-]+", "_", medication)

    cache_suffix = "all" if max_reviews is None else str(max_reviews)

    cache_path = (
        PROJECT_ROOT
        / "careinsight"
        / "results"
        / f"{safe_name}_{cache_suffix}_topics.json"
    )

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as file:
            print(f"Loading cached topic results: {cache_path.name}")
            return json.load(file)

    matches = reviews_df[
        reviews_df["drugName"]
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
            "message": "No matching patient reviews were found.",
        }

    total_reviews_available = len(matches)

    if max_reviews is not None:
        matches = matches.head(max_reviews)

    review_texts = [
        html.unescape(review).replace("\n", " ").strip()
        for review in matches["review"].astype(str).tolist()
    ]

    # Skip topic modeling when there are too few reviews.
    MIN_REVIEWS_FOR_TOPIC_MODELING = 20

    if len(review_texts) < MIN_REVIEWS_FOR_TOPIC_MODELING:
        result = {
            "medication": medication_name,
            "total_reviews_available": int(total_reviews_available),
            "reviews_analyzed": len(review_texts),
            "number_of_topics": 0,
            "outlier_reviews": None,
            "topics": [],
            "model": "BERTopic with all-MiniLM-L6-v2",
            "note": (
                f"Topic modeling was skipped because only "
                f"{len(review_texts)} reviews were available. "
                f"At least {MIN_REVIEWS_FOR_TOPIC_MODELING} "
                "reviews are required."
            ),
        }

        with open(cache_path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False)

        print(
            f"Skipping topic modeling: only "
            f"{len(review_texts)} reviews available."
        )

        return result

    topic_model, topics, probabilities = extract_topics(review_texts)

    topic_info = get_topic_summary(topic_model)

    topic_results = []

    for _, row in topic_info.head(10).iterrows():
        topic_id = int(row["Topic"])

        keywords = topic_model.get_topic(topic_id) or []

        topic_results.append({
            "topic_id": topic_id,
            "topic_name": str(row["Name"]),
            "review_count": int(row["Count"]),
            "keywords": [
                word for word, score in keywords[:8]
            ],
        })

    outlier_count = int(sum(topic == -1 for topic in topics))

    result = {
        "medication": medication_name,
        "total_reviews_available": int(total_reviews_available),
        "reviews_analyzed": len(review_texts),
        "number_of_topics": int(len(topic_info)),
        "outlier_reviews": outlier_count,
        "topics": topic_results,
        "model": "BERTopic with all-MiniLM-L6-v2",
        "note": (
            "Topics represent recurring language patterns in "
            "patient-reported reviews. They are exploratory themes, "
            "not verified clinical findings."
        ),
    }

    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print(f"Saved topic results: {cache_path.name}")

    return result

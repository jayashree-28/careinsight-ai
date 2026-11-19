
from pathlib import Path

import pandas as pd


# Locate the project and dataset directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DATA_FILES = [
    DATA_DIR / "drugsComTrain_raw.csv",
    DATA_DIR / "drugsComTest_raw.csv",
]


def search_medication_reviews(
    medication_name: str,
    max_examples: int = 5,
) -> dict:
    """
    Retrieve patient reviews for a medication.

    Returns review counts, rating statistics, and
    a limited number of supporting review examples.
    """

    if not medication_name or not medication_name.strip():
        return {"error": "Please provide a medication name."}

    if not 1 <= max_examples <= 10:
        return {"error": "max_examples must be between 1 and 10."}

    # Load both dataset files
    frames = []

    for file_path in DATA_FILES:
        if not file_path.exists():
            return {"error": f"Dataset file not found: {file_path}"}

        df = pd.read_csv(file_path)
        frames.append(df)

    reviews = pd.concat(frames, ignore_index=True)

    # Match medication names without case sensitivity
    medication = medication_name.strip().casefold()

    matches = reviews[
        reviews["drugName"].astype(str).str.strip().str.casefold()
        == medication
    ].copy()

    if matches.empty:
        return {
            "medication": medication_name,
            "total_reviews": 0,
            "message": "No matching medication was found."
        }

    # Calculate descriptive statistics
    rating_counts = (
        matches["rating"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # Retrieve a limited number of actual patient reviews
    examples = matches.head(max_examples)

    example_records = []

    for _, row in examples.iterrows():
        example_records.append({
            "review_id": int(row["uniqueID"]),
            "condition": str(row["condition"]),
            "rating": int(row["rating"]),
            "date": str(row["date"]),
            "useful_count": int(row["usefulCount"]),
            "review": str(row["review"]),
        })

    return {
        "medication": medication_name,
        "total_reviews": int(len(matches)),
        "average_rating": round(float(matches["rating"].mean()), 2),
        "rating_distribution": {
            str(int(k)): int(v)
            for k, v in rating_counts.items()
        },
        "supporting_reviews": example_records,
        "data_sources": [
            file_path.name for file_path in DATA_FILES
        ],
        "note": (
            "These are patient-reported reviews and ratings. "
            "Ratings alone do not establish sentiment."
        ),
    }

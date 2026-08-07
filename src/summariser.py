# src/summariser.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd

def clean_text(text: str) -> str:
    """
    Clean HTML entities and extra whitespace from review text.
    """
    return (text.replace("&#039;", "'")
                .replace("&amp;", "&")
                .replace("&quot;", '"')
                .replace("\n", " ")
                .strip())


def load_summariser():
    """
    Load BART model directly for summarisation.
    """
    print("Loading summarisation model...")
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def summarise_general(reviews: list, tokenizer, model) -> str:
    """
    Summarise patient reviews using DistilBART.
    """
    # Combine and truncate reviews
    combined = " ".join(reviews)[:1024]

    # Tokenize
    inputs = tokenizer(
        combined,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    # Generate summary
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=50,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )
    return summary


def summarise_reviews(reviews: list, tokenizer, model) -> dict:
    """
    Run summarisation on patient reviews.
    MedGemma summary added in Phase 3.
    """
    print("\n📝 Running general model summarisation...")
    general_summary = summarise_general(reviews, tokenizer, model)

    return {
        "general_summary": general_summary,
        "medgemma_summary": "Coming in Phase 3..."
    }


if __name__ == "__main__":
    # Load model
    tokenizer, model = load_summariser()

    # Load sample reviews
    df = pd.read_csv("data/drugsComTrain_raw.csv",
                     sep=",",
                     on_bad_lines="skip",
                     encoding="utf-8")

    # Take 10 Depression reviews
    sample_reviews = (df[df["condition"] == "Depression"]
                      ["review"]
                      .dropna()
                      .head(10)
                      .tolist())

    # Clean them
    sample_reviews = [clean_text(r) for r in sample_reviews]

    print("=" * 60)
    print("CareInsight AI — Summarisation Test")
    print("=" * 60)
    print(f"\n📋 Summarising {len(sample_reviews)} Depression reviews...")
    print("\nSample input review:")
    print(sample_reviews[0][:200] + "...")

    # Run summarisation
    results = summarise_reviews(sample_reviews, tokenizer, model)

    print("\n" + "=" * 60)
    print("📄 GENERAL MODEL SUMMARY (DistilBART):")
    print("=" * 60)
    print(results["general_summary"])

    print("\n" + "=" * 60)
    print("🏥 MEDGEMMA SUMMARY:")
    print("=" * 60)
    print(results["medgemma_summary"])

    print("\n✅ Summarisation test complete!")
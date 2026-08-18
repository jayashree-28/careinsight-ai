# src/medgemma.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
from src.summariser import clean_text, load_summariser, summarise_general

# Use a medical-focused summarisation model
MEDICAL_MODEL = "Falconsai/medical_summarization"


def load_medical_model():
    """
    Load Falconsai medical summarisation model.
    Fine-tuned specifically on medical text summarisation.
    """
    print("Loading medical summarisation model...")
    tokenizer = AutoTokenizer.from_pretrained(MEDICAL_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MEDICAL_MODEL)
    print("✅ Medical model loaded successfully!")
    return tokenizer, model


def summarise_medical(reviews: list, tokenizer, model) -> str:
    combined = " ".join(reviews)[:1024]

    inputs = tokenizer(
        combined,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )
    
    # Remove duplicate sentences
    sentences = summary.split(".")
    seen = []
    unique = []
    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            seen.append(s)
            unique.append(s)
    
    return ". ".join(unique[:3]).strip()


def compare_models(reviews: list,
                   gen_tokenizer, gen_model,
                   med_tokenizer, med_model) -> dict:
    """
    Run both models on same reviews and return comparison.
    """
    print("\n📄 Running general model (DistilBART)...")
    general_summary = summarise_general(
        reviews, gen_tokenizer, gen_model
    )

    print("\n🏥 Running medical summarisation model...")
    medical_summary = summarise_medical(
        reviews, med_tokenizer, med_model
    )

    return {
        "general_summary": general_summary,
        "medgemma_summary": medical_summary
    }


if __name__ == "__main__":
    print("=" * 60)
    print("CareInsight AI — Medical vs General Model Comparison")
    print("=" * 60)

    # Load both models
    print("\nLoading DistilBART (general)...")
    gen_tokenizer, gen_model = load_summariser()

    print("\nLoading medical model...")
    med_tokenizer, med_model = load_medical_model()

    # Load dataset
    df = pd.read_csv("data/drugsComTrain_raw.csv",
                     sep=",",
                     on_bad_lines="skip",
                     encoding="utf-8")

    # Test on 3 conditions
    conditions = ["Depression", "Anxiety", "Pain"]

    for condition in conditions:
        reviews = (df[df["condition"] == condition]["review"]
                   .dropna()
                   .head(5)
                   .tolist())
        reviews = [clean_text(r) for r in reviews]

        print(f"\n{'=' * 60}")
        print(f"📋 Condition: {condition} ({len(reviews)} reviews)")
        print("=" * 60)

        results = compare_models(
            reviews,
            gen_tokenizer, gen_model,
            med_tokenizer, med_model
        )

        print(f"\n📄 GENERAL MODEL (DistilBART):")
        print(f"   {results['general_summary']}")

        print(f"\n🏥 MEDICAL MODEL (Falconsai):")
        print(f"   {results['medgemma_summary']}")

        print(f"\n{'—' * 60}")

    print("\n✅ Comparison complete!")# src/medgemma.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
from src.summariser import clean_text, load_summariser, summarise_general

# Use a medical-focused summarisation model
MEDICAL_MODEL = "Falconsai/medical_summarization"


def load_medical_model():
    """
    Load Falconsai medical summarisation model.
    Fine-tuned specifically on medical text summarisation.
    """
    print("Loading medical summarisation model...")
    tokenizer = AutoTokenizer.from_pretrained(MEDICAL_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MEDICAL_MODEL)
    print("✅ Medical model loaded successfully!")
    return tokenizer, model


def summarise_medical(reviews: list, tokenizer, model) -> str:
    """
    Generate medically-aware summary using
    Falconsai medical summarisation model.
    """
    combined = " ".join(reviews)[:1024]

    inputs = tokenizer(
        combined,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=200,
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


def compare_models(reviews: list,
                   gen_tokenizer, gen_model,
                   med_tokenizer, med_model) -> dict:
    """
    Run both models on same reviews and return comparison.
    """
    print("\n📄 Running general model (DistilBART)...")
    general_summary = summarise_general(
        reviews, gen_tokenizer, gen_model
    )

    print("\n🏥 Running medical summarisation model...")
    medical_summary = summarise_medical(
        reviews, med_tokenizer, med_model
    )

    return {
        "general_summary": general_summary,
        "medgemma_summary": medical_summary
    }


if __name__ == "__main__":
    print("=" * 60)
    print("CareInsight AI — Medical vs General Model Comparison")
    print("=" * 60)

    # Load both models
    print("\nLoading DistilBART (general)...")
    gen_tokenizer, gen_model = load_summariser()

    print("\nLoading medical model...")
    med_tokenizer, med_model = load_medical_model()

    # Load dataset
    df = pd.read_csv("data/drugsComTrain_raw.csv",
                     sep=",",
                     on_bad_lines="skip",
                     encoding="utf-8")

    # Test on 3 conditions
    conditions = ["Depression", "Anxiety", "Pain"]

    for condition in conditions:
        reviews = (df[df["condition"] == condition]["review"]
                   .dropna()
                   .head(5)
                   .tolist())
        reviews = [clean_text(r) for r in reviews]

        print(f"\n{'=' * 60}")
        print(f"📋 Condition: {condition} ({len(reviews)} reviews)")
        print("=" * 60)

        results = compare_models(
            reviews,
            gen_tokenizer, gen_model,
            med_tokenizer, med_model
        )

        print(f"\n📄 GENERAL MODEL (DistilBART):")
        print(f"   {results['general_summary']}")

        print(f"\n🏥 MEDICAL MODEL (Falconsai):")
        print(f"   {results['medgemma_summary']}")

        print(f"\n{'—' * 60}")

    print("\n✅ Comparison complete!")
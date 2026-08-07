# src/topics.py
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import plotly.express as px

def load_reviews(filepath: str, n_samples: int = 500) -> list:
    """
    Load and clean patient reviews from CSV.
    """
    df = pd.read_csv(filepath,
                     sep=",",
                     on_bad_lines="skip",
                     encoding="utf-8")
    
    # Drop nulls and clean HTML entities
    reviews = df["review"].dropna().head(n_samples).tolist()
    reviews = [r.replace("&#039;", "'")
                .replace("&amp;", "&")
                .replace("&quot;", '"')
                .strip() for r in reviews]
    return reviews


def extract_topics(reviews: list) -> tuple:
    """
    Extract topics from patient reviews using BERTopic.
    Returns topic model and topic assignments.
    """
    print(f"\n🔍 Analysing {len(reviews)} patient reviews...")
    
    # Load sentence transformer for embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Use CountVectorizer to remove stop words
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=2,
        ngram_range=(1, 2)
    )
    
    # Create BERTopic model with better settings
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        nr_topics="auto",
        min_topic_size=10,
        verbose=True
    )
    
    # Fit model and get topics
    topics, probs = topic_model.fit_transform(reviews)
    
    return topic_model, topics, probs

def get_topic_summary(topic_model: BERTopic) -> pd.DataFrame:
    """
    Get a clean summary of discovered topics.
    """
    topic_info = topic_model.get_topic_info()
    
    # Filter out outlier topic (-1)
    topic_info = topic_info[topic_info["Topic"] != -1]
    
    return topic_info


def plot_topics(topic_info: pd.DataFrame):
    """
    Plot topic distribution using Plotly.
    """
    fig = px.bar(
        topic_info.head(10),
        x="Count",
        y="Name",
        orientation="h",
        title="Top 10 Patient Feedback Topics",
        labels={"Count": "Number of Reviews", "Name": "Topic"},
        color="Count",
        color_continuous_scale="blues"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.show()


if __name__ == "__main__":
    # Load reviews
    reviews = load_reviews("data/drugsComTrain_raw.csv", n_samples=2000)    
    print("=" * 60)
    print("CareInsight AI — Topic Modelling Test")
    print("=" * 60)
    
    # Extract topics
    topic_model, topics, probs = extract_topics(reviews)
    
    # Get summary
    topic_info = get_topic_summary(topic_model)
    
    print("\n📊 Discovered Topics:")
    print("=" * 60)
    for _, row in topic_info.head(10).iterrows():
        print(f"\nTopic {row['Topic']}: {row['Name']}")
        print(f"Reviews in topic: {row['Count']}")
    
    # Plot
    plot_topics(topic_info)
    
    print("\n✅ Topic modelling complete!")
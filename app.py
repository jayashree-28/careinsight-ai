# app.py
# spaces is only available on Hugging Face Spaces — must be imported
# before torch/transformers when present
try:
    import spaces
    HF_SPACES = True
except ImportError:
    HF_SPACES = False

    class _DummySpaces:
        """Fallback so @spaces.GPU is a no-op when running locally."""
        @staticmethod
        def GPU(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    spaces = _DummySpaces()

import gradio as gr
import pandas as pd
import plotly.express as px

from src.sentiment import analyse_sentiment
from src.summariser import clean_text, load_summariser, summarise_general
from src.medgemma import load_medical_model, summarise_medical
from src.pipeline import extract_top_keywords

# ── Load models once at startup ──────────────────────────────
print("🚀 Loading CareInsight AI models...")
gen_tokenizer, gen_model = load_summariser()
med_tokenizer, med_model = load_medical_model()
print("✅ All models loaded — launching app!")


def create_topic_chart(topics: list, condition: str):
    """Create Plotly bar chart of topics."""
    fig = px.bar(
        x=topics,
        y=[1] * len(topics),
        labels={"x": "Topics", "y": "Relevance"},
        title=f"Key Topics — {condition}",
        color=topics,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        height=300
    )
    return fig


@spaces.GPU(duration=120)
def analyse_reviews(text_input: str, condition: str):
    """
    Main analysis function — sentiment, topics, and both summaries.
    """
    if not text_input or len(text_input.strip()) < 10:
        return (
            "⚠️ Please enter at least one patient review.",
            "", "", "", None
        )

    # Split input into individual reviews
    reviews = [r.strip() for r in text_input.strip().split("\n")
               if r.strip()]
    reviews = [clean_text(r) for r in reviews]

    # 1. Sentiment analysis
    combined = " ".join(reviews)[:512]
    sentiment = analyse_sentiment(combined)
    emoji = {"positive": "😊", "negative": "😟", "neutral": "😐"}
    sentiment_display = (
        f"{emoji.get(sentiment['label'], '😐')} "
        f"{sentiment['label'].upper()} "
        f"({sentiment['score'] * 100:.1f}% confidence)"
    )

    # 2. Topic extraction
    topics = extract_top_keywords(reviews)
    topics_display = "\n".join([f"• {t}" for t in topics])

    # 3. General summary
    general_summary = summarise_general(reviews, gen_tokenizer, gen_model)

    # 4. Medical summary
    medical_summary = summarise_medical(reviews, med_tokenizer, med_model)

    # 5. Topic chart
    topic_fig = create_topic_chart(topics, condition)

    return (
        sentiment_display,
        topics_display,
        general_summary,
        medical_summary,
        topic_fig
    )


def analyse_from_dataset(condition: str, n_reviews: int):
    """Load reviews directly from the dataset by condition."""
    try:
        df = pd.read_csv(
            "data/drugsComTrain_raw.csv",
            sep=",",
            on_bad_lines="skip",
            encoding="utf-8"
        )
        reviews = (df[df["condition"] == condition]["review"]
                   .dropna()
                   .head(n_reviews)
                   .tolist())

        if not reviews:
            return "No reviews found for this condition.", "", "", "", None

        return analyse_reviews("\n".join(reviews), condition)

    except FileNotFoundError:
        return (
            "⚠️ Dataset not found. Use the 'Analyse Your Text' tab instead.",
            "", "", "", None
        )
    except Exception as e:
        return f"Error loading dataset: {str(e)}", "", "", "", None


# ── Gradio UI ────────────────────────────────────────────────
with gr.Blocks(title="CareInsight AI", theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    # 🏥 CareInsight AI
    ### Patient Feedback Intelligence Platform
    *Analyse patient reviews using NLP — sentiment, topics, and AI-powered summaries*
    ---
    """)

    with gr.Tabs():

        # ── Tab 1: Manual Input ──────────────────────────────
        with gr.Tab("📝 Analyse Your Text"):
            gr.Markdown("### Paste patient reviews below (one per line)")

            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Patient Reviews",
                        placeholder="Paste patient reviews here, one per line...",
                        lines=8
                    )
                    condition_input = gr.Textbox(
                        label="Condition (optional)",
                        placeholder="e.g. Depression, Anxiety, Pain",
                        value="General"
                    )
                    analyse_btn = gr.Button(
                        "🔍 Analyse Reviews",
                        variant="primary"
                    )

                with gr.Column(scale=3):
                    sentiment_out = gr.Textbox(
                        label="😊 Sentiment Analysis",
                        interactive=False
                    )
                    topics_out = gr.Textbox(
                        label="🏷️ Key Topics",
                        interactive=False,
                        lines=5
                    )

            with gr.Row():
                general_out = gr.Textbox(
                    label="📄 General Model Summary (DistilBART)",
                    interactive=False,
                    lines=4
                )
                medical_out = gr.Textbox(
                    label="🏥 Medical Model Summary (Falconsai)",
                    interactive=False,
                    lines=4
                )

            topic_chart = gr.Plot(label="📊 Topic Distribution")

            analyse_btn.click(
                fn=analyse_reviews,
                inputs=[text_input, condition_input],
                outputs=[
                    sentiment_out,
                    topics_out,
                    general_out,
                    medical_out,
                    topic_chart
                ]
            )

            gr.Examples(
                examples=[
                    [
                        "I have been taking Zoloft for 3 months now.\n"
                        "The side effects were bad at first but improved over time.\n"
                        "I feel much better and less anxious overall.",
                        "Depression"
                    ],
                    [
                        "This medication caused severe nausea and dizziness.\n"
                        "I had to stop taking it after only two weeks.\n"
                        "My doctor switched me to something else.",
                        "Pain"
                    ],
                ],
                inputs=[text_input, condition_input]
            )

        # ── Tab 2: Dataset Analysis ──────────────────────────
        with gr.Tab("📊 Analyse by Condition"):
            gr.Markdown("### Analyse reviews from the UCI Drug Review dataset")

            with gr.Row():
                condition_select = gr.Dropdown(
                    label="Select Condition",
                    choices=[
                        "Depression", "Anxiety", "Pain",
                        "Birth Control", "Diabetes, Type 2",
                        "High Blood Pressure", "Insomnia"
                    ],
                    value="Depression"
                )
                n_reviews_slider = gr.Slider(
                    label="Number of Reviews",
                    minimum=3,
                    maximum=20,
                    value=5,
                    step=1
                )
                dataset_btn = gr.Button(
                    "🔍 Analyse Dataset",
                    variant="primary"
                )

            with gr.Row():
                sentiment_out2 = gr.Textbox(
                    label="😊 Sentiment",
                    interactive=False
                )
                topics_out2 = gr.Textbox(
                    label="🏷️ Topics",
                    interactive=False,
                    lines=5
                )

            with gr.Row():
                general_out2 = gr.Textbox(
                    label="📄 General Model (DistilBART)",
                    interactive=False,
                    lines=4
                )
                medical_out2 = gr.Textbox(
                    label="🏥 Medical Model (Falconsai)",
                    interactive=False,
                    lines=4
                )

            topic_chart2 = gr.Plot(label="📊 Topics")

            dataset_btn.click(
                fn=analyse_from_dataset,
                inputs=[condition_select, n_reviews_slider],
                outputs=[
                    sentiment_out2,
                    topics_out2,
                    general_out2,
                    medical_out2,
                    topic_chart2
                ]
            )

        # ── Tab 3: About ─────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About CareInsight AI

            **CareInsight AI** is an AI-powered patient feedback
            intelligence platform built for healthcare providers.

            ### Models Used
            | Component | Model | Purpose |
            |---|---|---|
            | Sentiment | RoBERTa | Classify patient sentiment |
            | Topics | BERTopic | Discover recurring themes |
            | General Summary | DistilBART | Surface-level summary |
            | Medical Summary | Falconsai Medical | Clinical summary |

            ### Tech Stack
            Python • Hugging Face Transformers • BERTopic •
            LangChain • Gradio • Plotly • pytest • Great Expectations

            ### Dataset
            UCI Drug Review Dataset — 161,297 real patient reviews

            ### Testing
            29 automated tests covering sentiment output schema,
            topic extraction, pipeline integration, and data quality
            validation.
            """)

# ── Launch ───────────────────────────────────────────────────
if __name__ == "__main__":
    app.launch()
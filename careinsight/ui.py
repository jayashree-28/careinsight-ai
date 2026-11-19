from pathlib import Path
from dotenv import load_dotenv
import re
import html
import uuid
from io import BytesIO
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Load API configuration before importing the ADK agent.
ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_FILE)

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from careinsight.agent import root_agent


# ==================================================
# ADK CONFIGURATION
# ==================================================

APP_NAME = "careinsight"
USER_ID = "careinsight_ui_user"

session_service = InMemorySessionService()
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)


# ==================================================
# ANALYSIS OPTIONS
# ==================================================

ANALYSIS_OPTIONS = {
    "Review Statistics": (
        "Retrieve review statistics, total reviews, average rating, "
        "rating distribution, and supporting review examples."
    ),
    "Medication Reviews + FDA Information": (
        "Retrieve patient medication reviews, review counts, average "
        "ratings, rating distribution, and supporting review examples. "
        "Also retrieve FDA medication labeling information using the "
        "MCP-connected FDA lookup tool. Present patient-reported "
        "experiences and FDA labeling in separate sections. Clearly "
        "identify the FDA source and do not provide medical advice."
    ),
    "Sentiment Analysis": (
        "Analyze positive, neutral, and negative sentiment in the "
        "medication reviews. Include counts and percentages."
    ),
    "Topic Analysis": (
        "Identify recurring topics and themes in the medication "
        "reviews. Include topic counts and keywords."
    ),
    "Comprehensive Report + Critic": (
        "Generate a comprehensive medication experience report with "
        "review statistics, sentiment, and topics. Then validate the "
        "evidence using the Critic Agent. Present both the report and "
        "validation results."
    ),
}


# ==================================================
# PATIENT REVIEW PRESENTATION
# ==================================================

def render_patient_report_html(report_text, evidence=None):
    """Render patient review statistics and examples from structured evidence.

    Markdown parsing remains as a fallback for analysis modes that do not
    populate the comprehensive report evidence in ADK session state.
    """
    text = report_text or ""
    data = dashboard_data_from_evidence(evidence) if evidence else extract_dashboard_data(text)

    total = data["total_reviews"]
    average = data["average_rating"]
    ratings = data["ratings"]

    total_html = f"{total:,}" if total is not None else "—"
    average_html = f"{average:.2f}<span>/10</span>" if average is not None else "—"

    rating_rows = ""
    if ratings:
        max_count = max(ratings.values()) or 1
        for rating in sorted(ratings, reverse=True):
            count = ratings[rating]
            width = count / max_count * 100
            rating_rows += f"""
                <div class="rating-row">
                    <span class="rating-label">{html.escape(str(rating))}</span>
                    <div class="rating-track">
                        <div class="rating-fill" style="width:{width:.1f}%"></div>
                    </div>
                    <span class="rating-count">{count:,}</span>
                </div>
            """
    else:
        rating_rows = '<p class="review-muted">Rating distribution was not included in this report.</p>'

    review_cards = []
    retrieval = (evidence or {}).get("retrieval_analysis", {}) if evidence else {}
    supporting_reviews = retrieval.get("supporting_reviews", []) or []

    if supporting_reviews:
        for review in supporting_reviews:
            condition = html.escape(str(review.get("condition") or "Condition not specified"))
            rating = review.get("rating")
            rating_text = html.escape(str(rating)) if rating is not None else "Not provided"
            quote = html.escape(str(review.get("review") or "").strip())
            review_id = review.get("review_id")
            review_meta = (
                f'<div class="review-evidence-note">Review ID: {html.escape(str(review_id))}</div>'
                if review_id is not None else ""
            )
            if not quote:
                continue
            review_cards.append(f"""
                <details class="patient-review-card">
                    <summary>
                        <span class="review-condition">{condition}</span>
                        <span class="review-rating">{rating_text}/10 rating</span>
                        <span class="review-expand">Read patient experience</span>
                    </summary>
                    <blockquote>{quote}</blockquote>
                    {review_meta}
                    <div class="review-evidence-note">
                        Individual patient-submitted experience; not a clinical finding.
                    </div>
                </details>
            """)
    else:
        # Fallback for older report text that includes numbered review examples.
        review_pattern = re.compile(
            r"(?ms)^\s*\d+\.\s*\*\*Condition:\*\*\s*(.*?)\s*"
            r"\|\s*\*\*Rating:\*\*\s*(\d+(?:\.\d+)?)\s*/\s*10\s*\n"
            r"(.*?)(?=^\s*\d+\.\s*\*\*Condition:\*\*|^\s*#{1,4}\s|\Z)"
        )
        for match in review_pattern.finditer(text):
            condition = html.escape(re.sub(r"\s+", " ", match.group(1)).strip())
            rating = html.escape(match.group(2).strip())
            quote_lines = []
            for line in match.group(3).splitlines():
                line = re.sub(r"^\s*>\s?", "", line).strip().strip("*").strip()
                if line:
                    quote_lines.append(line)
            quote = html.escape(re.sub(r"\s+", " ", " ".join(quote_lines)).strip().strip('"'))
            if quote:
                review_cards.append(f"""
                    <details class="patient-review-card">
                        <summary>
                            <span class="review-condition">{condition}</span>
                            <span class="review-rating">{rating}/10 rating</span>
                            <span class="review-expand">Read patient experience</span>
                        </summary>
                        <blockquote>{quote}</blockquote>
                        <div class="review-evidence-note">
                            Individual patient-submitted experience; not a clinical finding.
                        </div>
                    </details>
                """)

    review_cards_html = "\n".join(review_cards) if review_cards else (
        '<p class="review-muted">Supporting patient review examples were not available in this report.</p>'
    )

    return f"""
    <div class="patient-experience">
        <div class="patient-metrics">
            <div class="patient-metric">
                <div class="patient-metric-label">Patient reviews retrieved</div>
                <div class="patient-metric-value">{total_html}</div>
            </div>
            <div class="patient-metric">
                <div class="patient-metric-label">Average patient rating</div>
                <div class="patient-metric-value">{average_html}</div>
                <div class="patient-metric-caption">Patient-submitted scores</div>
            </div>
        </div>
        <section class="patient-section">
            <div class="patient-section-heading">Rating distribution</div>
            <div class="patient-section-caption">Number of patient reviews for each score</div>
            {rating_rows}
            <div class="patient-footnote">
                Ratings reflect individual patient-submitted scores and are not formal sentiment classifications.
            </div>
        </section>
        <section class="patient-section">
            <div class="patient-section-heading">Patient review highlights</div>
            <div class="patient-section-caption">Select a review to read the full patient-reported experience.</div>
            {review_cards_html}
        </section>
    </div>
    """


def dashboard_data_from_evidence(evidence):
    """Convert ADK's structured report evidence into dashboard chart data."""
    evidence = evidence or {}
    retrieval = evidence.get("retrieval_analysis") or {}
    sentiment_result = evidence.get("sentiment_analysis") or {}
    topic_result = evidence.get("topic_analysis") or {}

    ratings = {}
    for rating, count in (retrieval.get("rating_distribution") or {}).items():
        try:
            ratings[int(rating)] = int(count)
        except (TypeError, ValueError):
            continue

    sentiment = {}
    for label, item in (sentiment_result.get("sentiment_distribution") or {}).items():
        if isinstance(item, dict):
            count = item.get("count")
        else:
            count = item
        try:
            sentiment[str(label).capitalize()] = int(count)
        except (TypeError, ValueError):
            continue

    topics = {}
    for index, item in enumerate(topic_result.get("topics") or []):
        if not isinstance(item, dict):
            continue
        label = item.get("topic_name") or item.get("name") or f"Topic {item.get('topic_id', index)}"
        count = item.get("review_count", item.get("count"))
        try:
            topics[str(label)] = int(count)
        except (TypeError, ValueError):
            continue

    total = retrieval.get("total_reviews")
    average = retrieval.get("average_rating")
    try:
        total = int(total) if total is not None else None
    except (TypeError, ValueError):
        total = None
    try:
        average = float(average) if average is not None else None
    except (TypeError, ValueError):
        average = None

    return {
        "total_reviews": total,
        "average_rating": average,
        "sentiment": sentiment,
        "ratings": ratings,
        "topics": topics,
        "outliers": topic_result.get("outlier_reviews"),
    }


# ==================================================
# DASHBOARD DATA EXTRACTION
# ==================================================

def _number(value):
    """Convert a formatted number such as '1,234' to an integer."""
    try:
        return int(value.replace(",", "").strip())
    except (AttributeError, ValueError):
        return None



def extract_dashboard_data(report_text):
    """Extract dashboard metrics from the agent's Markdown report."""

    text = report_text or ""

    # Remove Markdown formatting.
    text = re.sub(r"\*{1,3}", "", text)
    text = text.replace("`", "")

    data = {
        "total_reviews": None,
        "average_rating": None,
        "sentiment": {},
        "ratings": {},
        "topics": {},
        "outliers": None,
    }

    # 1. Total reviews.
    match = re.search(
        r"(?:total\s+(?:number\s+of\s+)?(?:patient\s+)?reviews"
        r"|total\s+reviews\s+(?:retrieved|analyzed)"
        r"|reviews\s+analyzed"
        r"|number\s+of\s+reviews)"
        r"\s*:\s*([\d,]+)",
        text,
        re.IGNORECASE,
    )

    if match:
        data["total_reviews"] = _number(match.group(1))

    # 2. Average rating.
    match = re.search(
        r"(?:average|mean)\s+(?:review\s+)?rating"
        r"\s*:\s*(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )

    if match:
        data["average_rating"] = float(match.group(1))

    # 3. Sentiment counts.
    for label in ("Positive", "Neutral", "Negative"):
        match = re.search(
            rf"\b{label}(?:\s+Sentiment)?\s*:\s*([\d,]+)",
            text,
            re.IGNORECASE,
        )

        if match:
            count = _number(match.group(1))
            if count is not None:
                data["sentiment"][label] = count

    # 4. Rating distribution.
    # Supports Markdown tables and standard rating labels.

    for match in re.finditer(
        r"^\s*\|\s*\*{0,2}(\d{1,2})\*{0,2}\s*\|"
        r"\s*\*{0,2}([\d,]+)\*{0,2}\s*\|",
        text,
        re.MULTILINE,
    ):
        rating = int(match.group(1))
        count = _number(match.group(2))
        if 1 <= rating <= 10 and count is not None:
            data["ratings"][rating] = count

    # Supports:
    # 1 Star: 70
    # 10 Stars: 193
    # Rating 10: 193 reviews
    # 10/10: 193 reviews

    for match in re.finditer(
        r"^\s*[-*]?\s*\*{0,2}(?:Rating\s+)?(\d{1,2})"
        r"(?:\s*/\s*10|\s+Stars?)?"
        r"\s*\*{0,2}\s*:\s*\*{0,2}\s*([\d,]+)",
        text,
        re.IGNORECASE | re.MULTILINE,
    ):
        rating = int(match.group(1))
        count = _number(match.group(2))

        if 1 <= rating <= 10 and count is not None:
            data["ratings"][rating] = count

    # 5. Topic counts.
    # Supports inline counts and separate Assigned Reviews labels.

    topic_matches = list(re.finditer(
        r"^\s*[-*]?\s*(?:\d+\.\s*)?Topic\s+(\d+)\b",
        text,
        re.IGNORECASE | re.MULTILINE,
    ))

    for index, topic_match in enumerate(topic_matches):
        start = topic_match.start()

        end = (
            topic_matches[index + 1].start()
            if index + 1 < len(topic_matches)
            else len(text)
        )

        section = text[start:end]
        topic_number = topic_match.group(1)

        # First look for a count on the topic heading.
        heading = section.splitlines()[0]

        count_match = re.search(
            r":\s*([\d,]+)\s+reviews?\b",
            heading,
            re.IGNORECASE,
        )

        # Otherwise look for a separate count label.
        if not count_match:
            count_match = re.search(
                r"(?:Review\s+Count|Assigned\s+Reviews)"
                r"\s*:\s*([\d,]+)",
                section,
                re.IGNORECASE,
            )

        if count_match:
            count = _number(count_match.group(1))

            if count is not None:
                data["topics"][f"Topic {topic_number}"] = count

    # 6. Outlier count.
    match = re.search(
        r"Outlier\s+(?:Reviews?|Count)\s*:\s*([\d,]+)",
        text,
        re.IGNORECASE,
    )

    if match:
        data["outliers"] = _number(match.group(1))

    logger.debug("Extracted dashboard data: %s", data)

    return data


def figure_to_image(fig):
    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=140,
        bbox_inches="tight",
    )
    plt.close(fig)

    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image.load()
    buffer.close()

    return image

def make_placeholder_plot(message):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.axis("off")
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center",
        color="#718096", fontsize=11,
        wrap=True,
    )
    fig.tight_layout()
    return figure_to_image(fig)


def make_sentiment_plot(data):
    values = data["sentiment"]
    if not values:
        return make_placeholder_plot(
            "Sentiment chart will appear when sentiment counts are available."
        )

    labels = list(values.keys())
    counts = list(values.values())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.pie(
        counts,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.38, "edgecolor": "white"},
        textprops={"fontsize": 9},
    )
    ax.set_title("Patient Review Sentiment", fontsize=12, pad=12)
    fig.tight_layout()
    return figure_to_image(fig)


def make_rating_plot(data):
    values = data["ratings"]
    if not values:
        return make_placeholder_plot(
            "Rating distribution chart will appear when rating counts are available."
        )

    ratings = sorted(values)
    counts = [values[rating] for rating in ratings]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar([str(r) for r in ratings], counts)
    ax.set_title("Rating Distribution", fontsize=12, pad=12)
    ax.set_xlabel("Patient rating")
    ax.set_ylabel("Number of reviews")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return figure_to_image(fig)


def make_topic_plot(data):
    values = data["topics"]
    if not values:
        return make_placeholder_plot(
            "Topic chart will appear when topic counts are available."
        )

    labels = list(values.keys())
    counts = list(values.values())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(labels[::-1], counts[::-1])
    ax.set_title("Review Topic Distribution", fontsize=12, pad=12)
    ax.set_xlabel("Number of reviews")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return figure_to_image(fig)


def dashboard_updates(report_text, evidence=None):
    data = dashboard_data_from_evidence(evidence) if evidence else extract_dashboard_data(report_text)

    logger.debug("Dashboard topic data: %s", data["topics"])

    total = (
        f"{data['total_reviews']:,}"
        if data["total_reviews"] is not None
        else "—"
    )

    average = (
        f"{data['average_rating']:.2f} / 10"
        if data["average_rating"] is not None
        else "—"
    )

    # Generate dynamic AI dashboard insights.
    insight_lines = []

    # 1. Overall patient experience.
    if data["total_reviews"] is not None:
        insight_lines.append(
            f"### Overall Patient Experience\n"
            f"- **Reviews analyzed:** {data['total_reviews']:,}"
        )

    if data["average_rating"] is not None:
        insight_lines.append(
            f"- **Average rating:** {data['average_rating']:.2f} / 10"
        )

    # 2. High ratings (9 and 10).
    if data["ratings"]:
        high_ratings = (
            data["ratings"].get(9, 0)
            + data["ratings"].get(10, 0)
        )
        total_ratings = sum(data["ratings"].values())

        if total_ratings:
            high_percentage = high_ratings / total_ratings * 100

            insight_lines.append(
                f"- **High ratings (9–10):** "
                f"{high_ratings:,} ({high_percentage:.1f}%)"
            )

    # 3. Sentiment distribution.
    if data["sentiment"] and sum(data["sentiment"].values()) > 0:
        total_sentiment = sum(data["sentiment"].values())

        sentiment_lines = []

        for label, count in data["sentiment"].items():
            percentage = count / total_sentiment * 100

            sentiment_lines.append(
                f"- **{label}:** {count:,} ({percentage:.2f}%)"
            )

        insight_lines.append(
            "### Sentiment Distribution\n"
            + "\n".join(sentiment_lines)
        )

    # 4. Recurring themes from topic keywords.
    keywords = []
    if evidence:
        for topic in (evidence.get("topic_analysis") or {}).get("topics", []) or []:
            if isinstance(topic, dict):
                raw = topic.get("keywords", [])
                if isinstance(raw, str):
                    raw = raw.split(",")
                keywords.extend(str(term).strip() for term in (raw or []))
    else:
        keyword_matches = re.findall(
            r"(?:Key\s+)?Keywords\s*:\s*(.+)",
            report_text or "",
            re.IGNORECASE,
        )
        for match in keyword_matches:
            keywords.extend(term.strip().strip("*` ") for term in match.split(","))

    unique_keywords = list(dict.fromkeys(term for term in keywords if term))
    if unique_keywords:
        insight_lines.append(
            "### Recurring Patient Themes\n" + ", ".join(unique_keywords[:12])
        )

    insights = "\n\n".join(insight_lines)

    if not insights:
        insights = "Insights will appear when report metrics are available."

    return (
        total,
        average,
        make_sentiment_plot(data),
        make_rating_plot(data),
        make_topic_plot(data),
        insights,
    )


# ==================================================
# RESULT FORMATTING
# ==================================================

def split_fda_section(report_text):
    """Extract every FDA labeling block from a combined agent response.

    The workflow can emit FDA labeling more than once (for example, in both
    the assistant's final response and a later agent response). Extracting
    only the first occurrence can leave a duplicate FDA block in the Critic
    output. This function removes all FDA blocks while preserving the order
    and content of the non-FDA sections.
    """
    text = report_text or ""

    heading_pattern = re.compile(
        r"(?im)^[ \t]*(?P<heading>"
        r"(?:#{1,6}[ \t]*)?"
        r"(?:\*\*[ \t]*)?"
        r"(?:Part[ \t]+\d+[ \t]*:[ \t]*)?"
        r"(?:Official[ \t]+)?"
        r"FDA[ \t]+(?:Approved[ \t]+)?Medication[ \t]+Labeling"
        r"(?:[ \t]+Information)?\b"
        r".*?"
        r"(?:\*\*)?"
        r")[ \t]*$"
    )

    matches = list(heading_pattern.finditer(text))
    if not matches:
        return (
            text.strip(),
            "FDA medication labeling information was not included in this analysis.",
        )

    # These headings mark a transition out of FDA labeling even if the
    # generated Markdown uses a deeper heading level than the FDA title.
    non_fda_boundary = re.compile(
        r"(?im)^[ \t]*#{0,6}[ \t]*"
        r"(?:Evidence Validation(?: Results)?|"
        r"Evidence Validation Results \(Critic Agent\)|"
        r"Critic Agent Evidence Validation Report|"
        r"Comprehensive Medication Experience Report|"
        r"Patient Medication Experience Report)"
        r"[ \t]*#*[ \t]*$"
    )

    extracted = []
    removal_spans = []

    for index, match in enumerate(matches):
        heading = match.group("heading")
        level_match = re.match(r"^[ \t]*(#{1,6})", heading)
        heading_level = len(level_match.group(1)) if level_match else None
        content_start = match.start()

        # Keep a horizontal rule immediately before the FDA block with it.
        prefix = text[:content_start]
        separator_match = re.search(
            r"(?:^|\n)[ \t]*---[ \t]*(?:\n|$)[ \t]*$", prefix
        )
        if separator_match:
            content_start = separator_match.start()

        possible_ends = []

        # A subsequent heading at the same or higher Markdown level ends
        # this FDA block. Lower-level headings are FDA subsections.
        if heading_level is not None:
            next_heading = re.search(
                rf"(?m)^[ \t]*#{{1,{heading_level}}}[ \t]+",
                text[match.end():],
            )
            if next_heading:
                possible_ends.append(match.end() + next_heading.start())

        # Critic/report headings are boundaries regardless of heading level.
        boundary = non_fda_boundary.search(text, match.end())
        if boundary:
            possible_ends.append(boundary.start())

        # Another FDA heading starts a separate FDA block, not part of this one.
        if index + 1 < len(matches):
            possible_ends.append(matches[index + 1].start())

        section_end = min(possible_ends) if possible_ends else len(text)
        extracted.append(text[match.start():section_end].strip())
        removal_spans.append((content_start, section_end))

    # Merge overlapping spans (possible when separator handling touches
    # adjacent blocks), then remove from right to left to keep offsets stable.
    merged = []
    for span_start, span_end in sorted(removal_spans):
        if merged and span_start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span_end))
        else:
            merged.append((span_start, span_end))

    remaining = text
    for span_start, span_end in reversed(merged):
        remaining = remaining[:span_start] + remaining[span_end:]

    remaining = re.sub(r"\n[ \t]*---[ \t]*(?=\n|$)", "", remaining)
    remaining = re.sub(r"\n{3,}", "\n\n", remaining).strip()

    # Multiple agents may independently summarize the same FDA lookup. Do not
    # concatenate those summaries: that produces two full FDA reports in the
    # FDA Information tab. Keep the most complete block (usually the longest)
    # and remove the app footer if it was included in the assistant response.
    cleaned_sections = []
    for part in extracted:
        part = re.sub(
            r"(?im)^CareInsight AI\s*[·|].*$", "", part
        ).strip()
        if part:
            cleaned_sections.append(part)

    fda_report = max(cleaned_sections, key=len) if cleaned_sections else ""
    return remaining, fda_report or (
        "FDA medication labeling information was not included in this analysis."
    )

def format_analysis_output(result):
    """Split the agent response into patient, FDA, and validation sections.

    FDA content is extracted from the complete response first, so it remains
    separate even when the agent places it after the Critic Agent section.
    """
    # Extract FDA labeling before identifying the Critic section. FDA content
    # can appear before or after validation in the combined agent response.
    text_without_fda, fda_report = split_fda_section(result or "")

    critic_match = re.search(
        r"(?im)^\s*#{0,6}\s*(?:"
        r"Evidence Validation Results(?:\s*\(Critic Agent\))?|"
        r"Critic Agent Evidence Validation Report|"
        r"Evidence Validation(?:\s+Results)?"
        r")\s*#*\s*$",
        text_without_fda,
    )
    if not critic_match:
        # Fallback: locate the first explicit validation field even if the
        # agent omitted or reformatted the section heading.
        critic_match = re.search(
            r"(?im)^\s*[-*]?\s*\*{0,2}Validation Status:\*{0,2}",
            text_without_fda,
        )

    if critic_match:
        report = text_without_fda[:critic_match.start()].strip()
        critic_section = text_without_fda[critic_match.start():].strip()
    else:
        report = text_without_fda.strip()
        critic_section = ""

    patient_report = report

    if not critic_section:
        return (
            patient_report,
            fda_report,
            gr.update(value="", visible=False),
            "The Critic Agent did not return a recognizable validation section. "
            "Check the application terminal for the agent response diagnostics.",
        )

    status_match = re.search(
        r"(?im)^\s*[-*]?\s*\*{0,2}Validation Status:\*{0,2}\s*`?([A-Z_ ]+)`?",
        critic_section,
    )
    checks_match = re.search(
        r"(?im)^\s*[-*]?\s*\*{0,2}Checks Passed:\*{0,2}\s*(\d+\s*/\s*\d+)",
        critic_section,
    )
    status = status_match.group(1).strip().upper() if status_match else "COMPLETED"
    checks = re.sub(r"\s+", "", checks_match.group(1)) if checks_match else None

    if status == "APPROVED":
        status_text = "### Evidence Validation: APPROVED"
    elif status == "REVISION_REQUIRED":
        status_text = "### Evidence Validation: REVISION REQUIRED"
    else:
        status_text = f"### Evidence Validation: {status}"
    if checks:
        status_text += f" — {checks} checks passed"

    return (
        patient_report,
        fda_report,
        gr.update(value=status_text, visible=True),
        critic_section,
    )


# ==================================================
# ADK INTERACTION
# ==================================================

async def analyze_medication(medication_name, analysis_type, session_id):
    """Run the selected ADK workflow and update all dashboard sections."""
    placeholder = make_placeholder_plot("Charts will appear after a successful analysis.")

    def empty_result(message):
        return (
            message,
            "No FDA labeling section was returned for this request.",
            gr.update(value="", visible=False),
            "Evidence validation details will appear here when the Critic Agent runs.",
            "—", "—",
            make_placeholder_plot(message),
            make_placeholder_plot(message),
            make_placeholder_plot(message),
            "Insights will appear after analysis.",
            session_id,
        )

    if not medication_name or not medication_name.strip():
        return empty_result("Please enter a medication name.")
    if not analysis_type:
        return empty_result("Please select an analysis type.")

    medication_name = medication_name.strip()
    # Always create a fresh session for each medication analysis.
    # This prevents previous medication context and evidence
    # from carrying into a new request.
    session_id = str(uuid.uuid4())

    prompt = (
        f"Analyze {medication_name}. "
        f"{ANALYSIS_OPTIONS[analysis_type]} "
        f"Also retrieve FDA medication labeling information for "
        f"{medication_name} using the lookup_medication_info tool. "
        f"Include indications, warnings, adverse reactions, "
        f"drug interactions, the exact retrieved_at_utc timestamp from the "
        f"tool result (label it as the FDA data retrieval date/time in UTC), "
        f"and a clickable Markdown source link using the exact source_url "
        f"returned by the tool. Format the link as [FDA openFDA Drug Label API](URL). "
        f"Do not invent or substitute a retrieval date. "
        f"Keep FDA labeling information separate from patient-reported "
        f"experiences."
    )
    message = types.Content(
        role="user",
        parts=[types.Part(text=prompt)],
    )

    try:
        response_parts = []
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
        )
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=message,
        ):
            if not event.content:
                continue
            event_text = "\n".join(
                part.text for part in event.content.parts or [] if part.text
            ).strip()
            print(
                f"DEBUG | author={event.author} "
                f"| final={event.is_final_response()} "
                f"| text_length={len(event_text)}"
            )

            if "FDA Medication Labeling" in event_text:
                print("DEBUG | FDA CONTENT FOUND IN EVENT")
            if not event.is_final_response() or not event_text:
                continue

            # Collect known analysis agents, but also recognize Critic output
            # by its content in case the ADK author name differs after a transfer.
            is_critic_content = bool(re.search(
                r"(?im)(Evidence Validation Results|Critic Agent Evidence Validation Report|"
                r"Validation Status:|Checks Passed:)",
                event_text,
            ))
            if event.author in ("retrieval_agent", "report_agent", "critic_agent") or is_critic_content:
                if event_text not in response_parts:
                    response_parts.append(event_text)
            elif event.author == "careinsight_assistant":
                if event_text not in response_parts:
                    response_parts.append(event_text)

        result = "\n\n".join(response_parts)
        print("DEBUG | Collected response length:", len(result))
        print(
            "DEBUG | FDA heading in assembled response:",
            "FDA Medication Labeling" in result
        )
        print("DEBUG | Response preview:", result[:1000])
        logger.info(
            "Analysis response collected: %d sections; Critic markers present: %s",
            len(response_parts),
            bool(re.search(r"(?i)(Evidence Validation Results|Validation Status:|Checks Passed:)", result)),
        )
        if not result:
            result = "The agent did not return a text response."

        # ToolContext.state writes are persisted to the ADK session. Read the
        # structured evidence after the workflow has completed instead of
        # relying on the agent to reproduce every value in Markdown.
        completed_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
        )
        session_state_data = completed_session.state if completed_session else {}
        evidence = session_state_data.get("careinsight_evidence")
        logger.debug("Structured evidence available: %s", bool(evidence))

        if evidence:
            logger.debug("Evidence sections: %s", list(evidence.keys()))
            logger.debug(
                "Supporting review examples: %d",
                len(
                    (evidence.get("retrieval_analysis") or {}).get(
                        "supporting_reviews", []
                    ) or []
                ),
            )

        (
            patient_report,
            fda_report,
            critic_status_update,
            validation_details,
        ) = format_analysis_output(result)

        (
            total_reviews,
            average_rating,
            sentiment_plot,
            rating_plot,
            topic_plot,
            insights,
        ) = dashboard_updates(patient_report, evidence=evidence)

        return (
            render_patient_report_html(patient_report, evidence=evidence),
            fda_report,
            critic_status_update,
            validation_details,
            total_reviews,
            average_rating,
            sentiment_plot,
            rating_plot,
            topic_plot,
            insights,
            session_id,
        )

    except Exception:
        logger.exception("Medication analysis failed.")

        error_message = (
            '<div class="review-empty-state">'
            '<strong>Analysis could not be completed.</strong> '
            'Please check the medication name and try again. '
            'If the problem continues, please try again later.'
            '</div>'
        )

        return (
            error_message,
            "FDA information is unavailable because the analysis failed.",
            gr.update(value="", visible=False),
            "Evidence validation details are unavailable because analysis failed.",
            "—", "—",
            placeholder, placeholder, placeholder,
            "Insights are unavailable because analysis failed.",
            session_id,
        )


# ==================================================
# INLINE DASHBOARD ICONS
# ==================================================

def dashboard_icon(name, size=20):
    """Return a small, self-contained SVG icon (no external assets required)."""
    paths = {
        "overview": '<rect x="3" y="3" width="7" height="7" rx="1.3"/><rect x="14" y="3" width="7" height="7" rx="1.3"/><rect x="3" y="14" width="7" height="7" rx="1.3"/><rect x="14" y="14" width="7" height="7" rx="1.3"/>',
        "reviews": '<path d="M4 4.5A2.5 2.5 0 0 1 6.5 2h11A2.5 2.5 0 0 1 20 4.5v11a2.5 2.5 0 0 1-2.5 2.5H9l-5 4V4.5Z"/><path d="M8 8h8M8 12h6"/>',
        "fda": '<path d="M12 3 20 6v5c0 5-3.4 8.5-8 10-4.6-1.5-8-5-8-10V6l8-3Z"/><path d="M9 8h6M9 12h6M9 16h3"/>',
        "validation": '<path d="m5 12 4 4L19 6"/><circle cx="12" cy="12" r="9"/>',
        "about": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 7h.01"/>',
        "medicine": '<path d="m10.5 3.5 10 10a5 5 0 0 1-7 7l-10-10a5 5 0 0 1 7-7Z"/><path d="m8 6 10 10"/>',
        "sentiment": '<path d="M3 12h4l2-6 4 12 2-6h6"/>',
        "rating": '<path d="m12 3 2.7 5.5 6.1.9-4.4 4.3 1 6.1-5.4-2.9-5.4 2.9 1-6.1-4.4-4.3 6.1-.9L12 3Z"/>',
        "topics": '<path d="M4 5h16M4 12h10M4 19h7"/><circle cx="18" cy="12" r="2"/><circle cx="15" cy="19" r="2"/>',
        "insights": '<path d="M9 18h6M10 22h4M8.5 14.5A7 7 0 1 1 15.5 14.5c-.9.7-1.5 1.7-1.5 2.5h-4c0-.8-.6-1.8-1.5-2.5Z"/>',
    }
    body = paths.get(name, paths["about"])
    return (
        f'<svg class="ci-icon" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{body}</svg>'
    )


# ==================================================
# CUSTOM DESIGN
# ==================================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1440px !important;
    margin: auto !important;
    background: #f4f7fb !important;
}
.main { padding: 0 !important; }
.main-workspace {
    width: 100%;
    max-width: 1320px;
    margin: 0 auto;
    padding: 0 18px 28px;
}
.page-header { padding: 12px 4px 20px; }
.page-title {
    font-size: 29px;
    font-weight: 750;
    color: #173653;
    letter-spacing: -1px;
    margin-bottom: 5px;
}
.page-description { color: #718096; font-size: 14px; }
.brand-heading { display:flex; align-items:center; gap:13px; }
.brand-mark { width:42px; height:42px; display:grid; place-items:center; flex:0 0 42px;
    border-radius:13px; color:#fff; background:linear-gradient(145deg,#2878b8,#174b7a);
    box-shadow:0 5px 14px rgba(23,75,122,.18); }
.brand-mark svg { width:24px; height:24px; }
.ci-section-heading { display:flex; align-items:center; gap:9px; color:#193b5a;
    font-size:17px; font-weight:700; margin:7px 0 12px; }
.ci-icon { display:inline-block; flex:0 0 auto; vertical-align:middle; color:#2878b8; }
.ci-metric-label { display:flex; align-items:center; gap:7px; color:#718096;
    font-size:12px; margin-bottom:8px; }
.ci-tab-intro { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.ci-tab-intro .ci-icon { width:23px; height:23px; }
.ci-tab-intro h3 { margin:0; color:#193b5a; font-size:19px; }

.panel {
    background: white;
    border: 1px solid #e6ebf2;
    border-radius: 13px;
    padding: 18px;
    box-shadow: 0 3px 12px rgba(25, 45, 75, 0.035);
}
.panel-heading {
    font-size: 17px;
    font-weight: 700;
    color: #193b5a;
    margin-bottom: 5px;
}
.panel-description {
    color: #8190a3;
    font-size: 12px;
    margin-bottom: 14px;
}
.primary-button {
    background: #1765a5 !important;
    border: none !important;
    color: white !important;
    font-weight: 650 !important;
}
.metric-card {
    background: white;
    border: 1px solid #e6ebf2;
    border-radius: 13px;
    padding: 17px 20px;
    min-height: 105px;
}
.metric-label {
    color: #718096;
    font-size: 12px;
    margin-bottom: 8px;
}
.metric-value {
    color: #193b5a;
    font-size: 29px;
    font-weight: 750;
}
.results-panel {
    background: white;
    border: 1px solid #e6ebf2;
    border-radius: 13px;
    padding: 22px;
}
.results-panel h1, .results-panel h2, .results-panel h3 {
    color: #193b5a;
}
.results-panel p, .results-panel li { line-height: 1.7; }
.results-panel .tab-nav {
    border-bottom: 1px solid #e6ebf2;
    gap: 8px;
}
.results-panel button[role="tab"] {
    border-radius: 9px 9px 0 0 !important;
    font-weight: 650 !important;
}
.results-panel .tabitem {
    padding: 20px 8px 8px;
    min-height: 900px;
    height: auto !important;
    max-height: none !important;
    overflow-y: visible !important;
    box-sizing: border-box;
}
.chart-panel {
    min-height: 390px;
    height: 390px;
    display: flex;
    flex-direction: column;
}
.chart-panel .chart-image,
.chart-panel .chart-image > div {
    height: 300px !important;
    min-height: 300px !important;
}
.chart-panel img {
    object-fit: contain !important;
}
@media (max-width: 760px) {
    .results-panel .tabitem {
        min-height: 900px;
        height: auto !important;
        max-height: none !important;
        overflow-y: visible !important;
    }
    .chart-panel { height: 360px; min-height: 360px; }
}
.report-content {
    line-height: 1.75;
    color: #34465a;
}
.report-content h1, .report-content h2, .report-content h3 {
    margin-top: 1.2em;
}

.patient-experience { color: #34465a; }
.patient-metrics {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
    margin: 18px 0 22px;
}
.patient-metric {
    background: #f7faff;
    border: 1px solid #e4edf7;
    border-radius: 12px;
    padding: 17px 19px;
}
.patient-metric-label, .patient-metric-caption,
.patient-section-caption, .patient-footnote, .review-evidence-note {
    color: #718096;
    font-size: 12px;
}
.patient-metric-value {
    color: #193b5a;
    font-size: 30px;
    font-weight: 750;
    margin-top: 5px;
}
.patient-metric-value span {
    color: #8190a3;
    font-size: 15px;
    font-weight: 500;
    margin-left: 3px;
}
.patient-section {
    border-top: 1px solid #e8edf3;
    padding-top: 18px;
    margin-top: 18px;
}
.patient-section-heading {
    color: #193b5a;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 3px;
}
.patient-section-caption { margin-bottom: 15px; }
.rating-row {
    display: grid;
    grid-template-columns: 22px minmax(60px, 1fr) 34px;
    align-items: center;
    gap: 10px;
    margin: 9px 0;
}
.rating-label, .rating-count {
    color: #53677d;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
}
.rating-count { text-align: right; }
.rating-track {
    background: #edf2f7;
    border-radius: 99px;
    height: 9px;
    overflow: hidden;
}
.rating-fill {
    background: #3278b9;
    border-radius: 99px;
    height: 100%;
}
.patient-footnote { margin-top: 14px; line-height: 1.5; }
.patient-review-card {
    background: #fbfcfe;
    border: 1px solid #e5ebf2;
    border-radius: 11px;
    margin: 10px 0;
    padding: 13px 15px;
}
.patient-review-card summary {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 9px 12px;
    cursor: pointer;
    list-style: none;
}
.patient-review-card summary::-webkit-details-marker { display: none; }
.review-condition {
    color: #193b5a;
    font-weight: 700;
}
.review-rating {
    color: #1765a5;
    background: #eaf3fb;
    border-radius: 20px;
    padding: 4px 9px;
    font-size: 11px;
    font-weight: 650;
}
.review-expand {
    color: #718096;
    font-size: 11px;
    margin-left: auto;
}
.patient-review-card blockquote {
    border-left: 3px solid #8bb5d9;
    color: #42566c;
    line-height: 1.7;
    margin: 13px 0 9px;
    padding: 2px 0 2px 13px;
}
.review-evidence-note { line-height: 1.5; }
.review-muted, .review-empty-state {
    color: #718096;
    background: #f7faff;
    border: 1px dashed #d7e2ee;
    border-radius: 10px;
    padding: 18px;
    font-size: 13px;
}
@media (max-width: 640px) {
    .patient-metrics { grid-template-columns: 1fr; }
    .review-expand { width: 100%; margin-left: 0; }
}

.critic-status {
    border-radius: 9px;
    padding: 10px 14px;
    margin: 10px 0 16px;
}
.disclaimer {
    background: #edf4fa;
    border: 1px solid #dce8f2;
    border-radius: 9px;
    padding: 12px 15px;
    color: #526a81;
    font-size: 11px;
    line-height: 1.6;
}
/* Hide Gradio's built-in footer while keeping the CareInsight footer. */
footer,
.gradio-container footer,
.gradio-container [data-testid="footer"],
.gradio-container .built-with {
    display: none !important;
}

.app-footer {
    color: #8a98a9;
    text-align: center;
    font-size: 11px;
    padding: 18px;
}
"""


# ==================================================
# GRADIO APPLICATION
# ==================================================

with gr.Blocks(
    title="CareInsight AI | Medication Research",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="teal",
        neutral_hue="slate",
    ),
    css=CUSTOM_CSS,
) as demo:

    session_state = gr.State(value=None)

    with gr.Column(elem_classes=["main-workspace"]):
        gr.HTML(
            """
            <div class="page-header">
                <div class="brand-heading">
                    <div class="brand-mark">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                            <path d="M12 3v18M3 12h18"/><circle cx="12" cy="12" r="9"/>
                        </svg>
                    </div>
                    <div>
                        <div class="page-title">CareInsight AI</div>
                        <div class="page-description">Patient medication experience research and evidence intelligence.</div>
                    </div>
                </div>
            </div>
            """
        )

        # Medication Research remains the primary action area at the top.
        with gr.Group(elem_id="medication-research-section", elem_classes=["panel"]):
            gr.HTML(
                f"""
                <div class="panel-heading ci-section-heading">{dashboard_icon("medicine")} Medication Research</div>
                <div class="panel-description">
                    Select a medication and choose the analysis to run. Results will
                    appear in the tabs below.
                </div>
                """
            )
            with gr.Row():
                medication_input = gr.Textbox(
                    label="Medication Name",
                    placeholder="e.g. Mirtazapine",
                    value="Mirtazapine",
                    scale=2,
                )
                analysis_type = gr.Dropdown(
                    choices=list(ANALYSIS_OPTIONS.keys()),
                    value="Comprehensive Report + Critic",
                    label="Analysis Type",
                    scale=2,
                )
            analyze_button = gr.Button(
                "Run Analysis",
                variant="primary",
                elem_classes=["primary-button"],
            )

        # All information is organized below the research form. Selecting a tab
        # displays that section without requiring a left-hand navigation menu.
        with gr.Group(elem_classes=["results-panel"]):
            with gr.Tabs():
                with gr.Tab("▦  Overview"):
                    gr.HTML(f'<div class="ci-tab-intro">{dashboard_icon("overview", 23)}<h3>Patient Experience Overview</h3></div>')
                    gr.Markdown(
                        "A snapshot of review volume, ratings, sentiment, and recurring themes."
                    )
                    with gr.Row():
                        with gr.Column(elem_classes=["metric-card"]):
                            gr.HTML(f'<div class="ci-metric-label">{dashboard_icon("reviews", 16)} Total Reviews</div>')
                            total_reviews_output = gr.Markdown(
                                "—", elem_classes=["metric-value"]
                            )
                        with gr.Column(elem_classes=["metric-card"]):
                            gr.HTML(f'<div class="ci-metric-label">{dashboard_icon("rating", 16)} Average Rating</div>')
                            average_rating_output = gr.Markdown(
                                "—", elem_classes=["metric-value"]
                            )
                    gr.HTML(f'<div class="ci-section-heading">{dashboard_icon("insights")} Patient Experience Analytics</div>')
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, elem_classes=["panel", "chart-panel"]):
                            gr.HTML(f'<div class="ci-section-heading">{dashboard_icon("sentiment", 18)} Sentiment Analysis</div>')
                            sentiment_plot = gr.Image(
                                value=make_placeholder_plot(
                                    "Run an analysis to populate sentiment results."
                                ),
                                show_label=False,
                                height=300,
                                elem_classes=["chart-image"],
                            )
                        with gr.Column(scale=1, elem_classes=["panel", "chart-panel"]):
                            gr.HTML(f'<div class="ci-section-heading">{dashboard_icon("rating", 18)} Rating Distribution</div>')
                            rating_plot = gr.Image(
                                value=make_placeholder_plot(
                                    "Run an analysis to populate rating results."
                                ),
                                show_label=False,
                                height=300,
                                elem_classes=["chart-image"],
                            )
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, elem_classes=["panel", "chart-panel"]):
                            gr.HTML(f'<div class="ci-section-heading">{dashboard_icon("topics", 18)} Topic Analysis</div>')
                            topic_plot = gr.Image(
                                value=make_placeholder_plot(
                                    "Run an analysis to populate topic results."
                                ),
                                show_label=False,
                                height=300,
                                elem_classes=["chart-image"],
                            )

                with gr.Tab("✦  AI Insights"):
                    gr.HTML(f'<div class="ci-tab-intro">{dashboard_icon("insights", 23)}<h3>AI-Generated Insights</h3></div>')
                    gr.Markdown(
                        "Interpretation of the available patient review statistics, sentiment patterns, and recurring themes."
                    )
                    with gr.Group(elem_classes=["panel"]):
                        insights_output = gr.Markdown(
                            "Insights will appear after analysis."
                        )

                with gr.Tab("▤  Patient Reviews"):
                    gr.HTML(f'<div class="ci-tab-intro">{dashboard_icon("reviews", 23)}<h3>Patient Medication Experience</h3></div>')
                    gr.Markdown(
                        "Patient-reported experiences and review evidence retrieved "
                        "for the selected medication."
                    )
                    patient_report_output = gr.HTML(
                        value=(
                            '<div class="review-empty-state">'
                            'Run an analysis to view patient review statistics '
                            'and supporting patient experiences.'
                            '</div>'
                        ),
                    )

                with gr.Tab("◇  FDA Information"):
                    gr.HTML(f'<div class="ci-tab-intro">{dashboard_icon("fda", 23)}<h3>FDA Medication Labeling</h3></div>')
                    gr.Markdown(
                        "Regulatory labeling information retrieved through the FDA "
                        "lookup tool. This is separate from patient reports and is "
                        "not medical advice."
                    )
                    fda_report_output = gr.Markdown(
                        value=(
                            "Select **Medication Reviews + FDA Information** "
                            "and run an analysis to retrieve FDA labeling."
                        ),
                        elem_classes=["report-content"],
                    )

                with gr.Tab("✓  Evidence Validation"):
                    critic_status = gr.Markdown(
                        value="Validation results will appear when the Critic Agent runs.",
                        visible=False,
                        elem_classes=["critic-status"],
                    )
                    gr.Markdown(
                        "The Critic Agent checks whether report claims are supported "
                        "by the retrieved evidence."
                    )
                    validation_details = gr.Markdown(
                        value="Evidence validation details will appear here when the Critic Agent runs.",
                        elem_classes=["report-content"],
                    )

                with gr.Tab("ⓘ  About CareInsight"):
                    gr.HTML(f'<div class="ci-tab-intro">{dashboard_icon("about", 23)}<h3>About CareInsight AI</h3></div>')
                    gr.Markdown(
                        "CareInsight AI analyzes public, patient-reported medication "
                        "reviews using retrieval, sentiment analysis, topic modeling, "
                        "and evidence validation."
                    )
                    gr.HTML(
                        """
                        <div class="disclaimer">
                            <strong>Important Information:</strong>
                            CareInsight AI analyzes public, patient-reported
                            medication reviews. Its outputs reflect subjective
                            experiences and machine-learning patterns, not verified
                            clinical findings or medical advice. Do not use this
                            application to make medication or treatment decisions.
                        </div>
                        """
                    )

        gr.HTML(
            """
            <div class="app-footer">
                CareInsight AI · Powered by Google ADK, Gemini, RoBERTa, and BERTopic
            </div>
            """
        )

    analyze_button.click(
        fn=analyze_medication,
        inputs=[
            medication_input,
            analysis_type,
            session_state,
        ],
        outputs=[
            patient_report_output,
            fda_report_output,
            critic_status,
            validation_details,
            total_reviews_output,
            average_rating_output,
            sentiment_plot,
            rating_plot,
            topic_plot,
            insights_output,
            session_state,
        ],
        show_progress="full",
        concurrency_limit=1,
    )


# ==================================================
# LAUNCH
# ==================================================
if __name__ == "__main__":
    demo.queue().launch()
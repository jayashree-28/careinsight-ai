# CareInsight AI — Patient Medication Experience Intelligence Platform

CareInsight AI is a multi-agent AI application that analyzes patient-reported medication reviews alongside official FDA medication labeling. It combines information retrieval, sentiment analysis, topic modeling, report generation, and evidence validation to produce structured medication experience reports.

The project demonstrates how multi-agent orchestration, external tools, NLP models, and automated testing can work together in a healthcare-related AI application.

## Project Overview

CareInsight AI brings together two distinct evidence sources:

- **Patient-reported information:** Public medication reviews describing individual experiences.
- **Official medication information:** FDA drug labeling retrieved through the openFDA API.

The application keeps these sources separate. Patient reviews represent individual experiences and are not proof that a medication caused a particular outcome.

## Key Features and Technologies

| Feature | Technology | Purpose |
|---|---|---|
| FDA medication retrieval | openFDA API + MCP | Retrieve official medication labeling |
| Agent orchestration | Google ADK | Coordinate the multi-agent workflow |
| Sentiment analysis | CardiffNLP RoBERTa | Classify review sentiment |
| Topic modeling | BERTopic | Discover recurring themes in reviews |
| Text embeddings | Sentence Transformers | Represent reviews for topic modeling |
| Report generation | Google Gemini | Generate structured analysis reports |
| Evidence validation | Critic Agent | Check report consistency and evidence support |
| Interactive interface | Gradio | Provide browser-based access |
| Data quality | Great Expectations | Validate dataset quality |
| Automated testing | pytest | Test application components |

## Multi-Agent Architecture

The medication analysis workflow uses a sequential agent architecture.

### 1. Retrieval Agent
- Retrieves patient medication reviews.
- Calls the medication information MCP tool.
- Retrieves official FDA labeling through openFDA.

### 2. Sentiment Agent
- Analyzes patient review sentiment.
- Reports positive, negative, and neutral sentiment patterns.

### 3. Topic Agent
- Uses BERTopic to identify recurring themes.
- Organizes reviews into topic groups.

### 4. Report Agent
- Combines the analysis results.
- Produces a structured medication experience report.
- Distinguishes FDA labeling from patient-reported experiences.

### 5. Critic Agent
- Validates report consistency and evidence references.
- Checks structural and numerical consistency.
- Reports validation errors and warnings.

The root agent delegates the medication analysis task to the sequential workflow.

## MCP Integration

The Retrieval Agent uses a Model Context Protocol (MCP) server to access medication information.

The MCP server:
- Calls the openFDA Drug Label API.
- Retrieves medication labeling information.
- Returns fields such as indications, warnings, adverse reactions, and interactions.
- Handles API errors and missing results.

The MCP server is started automatically by the Retrieval Agent through a standard input/output connection. It does not need to be started manually.

## Application Interfaces

CareInsight AI includes three ways to use the project.

| Application | Command | Description |
|---|---|---|
| ADK command-line workflow | `uv run adk run careinsight` | Run the multi-agent medication analysis |
| ADK Gradio interface | `uv run python -m careinsight.ui` | Use the multi-agent workflow through a browser |
| Standalone NLP interface | `uv run python app.py` | Analyze reviews using the original NLP pipeline |

The standalone NLP interface is separate from the Google ADK multi-agent workflow. These interfaces serve different purposes.

## Example Results

The following results were produced by the multi-agent workflow.

### Mirtazapine

| Metric | Result |
|---|---:|
| Patient reviews analyzed | 727 |
| Average rating | 7.18 / 10 |
| Positive sentiment | 305 (41.95%) |
| Negative sentiment | 280 (38.51%) |
| Neutral sentiment | 142 (19.53%) |
| Topics identified | 4 |
| Topic outliers | 46 |
| Critic validation | APPROVED |
| Validation checks passed | 14 / 14 |

### Sertraline

| Metric | Result |
|---|---:|
| Patient reviews analyzed | 1,868 |
| Average rating | 7.45 / 10 |
| Negative sentiment | 837 (44.81%) |
| Positive sentiment | 734 (39.29%) |
| Neutral sentiment | 297 (15.90%) |
| Topics identified | 4 |
| Topic outliers | 0 |
| Critic validation | APPROVED |
| Validation checks passed | 14 / 14 |

These examples demonstrate that numerical ratings and text-based sentiment are different measures. For example, Sertraline had an average rating of 7.45/10 while 44.81% of analyzed reviews were classified as negative.

The Critic Agent's approval indicates that the configured validation checks passed. It does not establish clinical accuracy or independently verify every medical claim.

## Dataset

The project uses the Drugs.com patient medication review dataset, distributed through Kaggle.

Expected local dataset files:

```text
data/
├── drugsComTrain_raw.csv
└── drugsComTest_raw.csv
```

Obtain the dataset from the Kaggle source used for this project and place the files in the `data/` directory.

The dataset contains patient-reported reviews and associated information such as medication name, medical condition, review text, rating, date, and useful count.

**Dataset availability:** The dataset files are expected locally and are not assumed to be included in this repository. Check the dataset's license and terms before redistributing it.

The multi-agent workflow and standalone NLP interface use different analysis paths. Refer to the [User Guide](docs/USER_GUIDE.md) for details.

## Project Structure

```text
careinsight-ai/
├── careinsight/
│   ├── results/
│   ├── agent.py
│   ├── retrieval_agent.py
│   ├── medication_mcp_server.py
│   ├── sentiment_agent.py
│   ├── topic_agent.py
│   ├── report_agent.py
│   ├── critic_agent.py
│   ├── critic_tools.py
│   ├── retrieval.py
│   ├── sentiment_tools.py
│   ├── topic_tools.py
│   ├── report_tools.py
│   └── ui.py
├── src/
│   ├── sentiment.py
│   ├── topics.py
│   ├── summariser.py
│   ├── medgemma.py
│   └── pipeline.py
├── rag/
├── data/
├── tests/
├── gx/
├── notebooks/
├── docs/
│   └── USER_GUIDE.md
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Configuration

### Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Google Gemini API key
- Dependencies listed in `requirements.txt`
- Dataset files in the `data/` directory, when required by the selected application path

The tested environment included:

| Package | Tested version |
|---|---:|
| google-adk | 2.14.0 |
| google-genai | 2.34.0 |
| mcp | 2.3.0 |

Make sure these packages are included in `requirements.txt`:

```text
google-adk==2.14.0
google-genai==2.34.0
mcp==2.3.0
```

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd careinsight-ai
```

Replace the placeholder with your actual repository URL.

### 2. Create and activate a virtual environment

```bash
uv venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the `careinsight/` directory:

```env
GOOGLE_API_KEY=your_google_api_key
```

Do not commit your actual `.env` file or API key to GitHub.

### 5. Run the application

Run the multi-agent workflow through the ADK command line:

```bash
uv run adk run careinsight
```

Or launch the multi-agent Gradio interface:

```bash
uv run python -m careinsight.ui
```

The Retrieval Agent automatically starts the MCP server when needed.

To run the standalone NLP interface:

```bash
uv run python app.py
```

For additional setup and configuration details, see [docs/USER_GUIDE.md](docs/USER_GUIDE.md).

## Running Tests

Run the test suite:

```bash
uv run pytest tests/ -v
```

**Latest recorded test result: 48 passed.**

The tests cover:
- Critic Agent validation
- Dataset and data quality checks
- NLP pipeline behavior
- Sentiment analysis
- Topic modeling

## Important Limitations

- Patient reviews are subjective and may contain incomplete or inaccurate information.
- Sentiment classifications represent model predictions, not verified clinical assessments.
- Topic modeling identifies patterns in text; topics are not established medical diagnoses.
- FDA labeling and patient reviews are different evidence sources and should not be treated as interchangeable.
- The Critic Agent validates configured report consistency and evidence requirements; it does not guarantee clinical correctness.
- The application does not establish medication causality and is not intended to provide medical advice.
- Some topic results may be loaded from cached analysis results rather than freshly trained during every run.

## Learning Outcomes

This project demonstrates practical experience with:

- Multi-agent system design and orchestration.
- Google ADK and Gemini integration.
- Model Context Protocol (MCP) and external API integration.
- NLP sentiment analysis and topic modeling.
- Evidence-aware report generation.
- Critic Agent design and validation.
- Automated testing and data quality validation.
- Gradio application development.

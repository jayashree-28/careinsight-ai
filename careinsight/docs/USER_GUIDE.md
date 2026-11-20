# CareInsight AI --- User Guide

This guide explains how to configure, install, run, and test CareInsight
AI.

## 1. Prerequisites

-   Python 3.12 (tested with Python 3.12.15)
-   Git
-   `uv` (for the commands shown below)
-   Google Gemini API key
-   Project dependencies
-   Patient review dataset, when required by the selected application

The tested environment uses:

  Package          Version
  -------------- ---------
  Python           3.12.15
  Google ADK        2.14.0
  Google GenAI      2.34.0
  MCP                2.3.0

## 2. Clone the Repository

``` bash
git clone https://github.com/jayashree-28/careinsight-ai.git
cd careinsight-ai
```

## 3. Create the Python Environment and Install Dependencies

The project uses a `requirements.txt` file with pinned dependency
versions.

### Option A: Using `uv`

Create and activate the virtual environment:

``` bash
uv venv
source .venv/bin/activate
```

Install all dependencies:

``` bash
uv pip install -r requirements.txt
```

### Option B: Using Python `venv` and `pip`

Create and activate the virtual environment:

``` bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Upgrade pip and install all dependencies:

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file already includes the pinned versions of
Google ADK, Google GenAI, and MCP. They do not need to be installed
separately.

## 4. Configure API Credentials

The ADK Gradio interface loads environment variables from:

``` text
careinsight/.env
```

Create this file if it does not exist.

Add your Google Gemini API key:

``` env
GOOGLE_API_KEY=your_google_api_key
```

Replace the placeholder with your own key.

**Security notes:**

-   Never commit `.env` to Git.
-   Never share your API key in screenshots, logs, or documentation.
-   Do not upload credentials to Hugging Face Spaces or GitHub as
    ordinary source files.
-   Use the hosting platform's secret-management settings when
    deploying.

The ADK command-line workflow also needs access to the Google API
credentials. Confirm that the credentials are available in the
environment used to launch ADK.

## 5. Prepare the Dataset

The project expects the following files for the dataset-dependent
functionality:

``` text
data/
├── drugsComTrain_raw.csv
└── drugsComTest_raw.csv
```

Place the dataset files in the `data/` directory.

The standalone NLP interface uses the training dataset for
condition-based review analysis. The multi-agent workflow uses its own
review retrieval implementation.

The dataset is not assumed to be included in this repository. Obtain it
from the Kaggle source used for this project and follow the dataset's
license and terms.

## 6. Run the Applications

CareInsight AI has three application entry points.

### Option A: ADK Command-Line Workflow

Run from the project root:

``` bash
uv run adk run careinsight
```

This launches the multi-agent medication analysis workflow.

The workflow includes:

1.  Retrieval Agent
2.  Sentiment Agent
3.  Topic Agent
4.  Report Agent
5.  Critic Agent

Enter a medication name when prompted.

The Retrieval Agent retrieves patient reviews and uses the MCP
medication information tool to retrieve FDA labeling.

The Critic Agent validates the generated report before the final result
is returned.

### Option B: ADK Gradio Interface

Run:

``` bash
uv run python -m careinsight.ui
```

This launches the browser-based interface for the multi-agent workflow.

The interface provides analysis options, patient experience statistics,
FDA information, report output, and evidence validation results.

Open the local address printed in the terminal.

The Gradio interface uses the ADK Runner to execute the root agent.

### Option C: Standalone NLP Gradio Interface

Run:

``` bash
uv run python app.py
```

This launches the original NLP review-analysis application.

It supports:

-   Entering patient reviews manually.
-   Analyzing reviews by medical condition using the dataset.
-   Sentiment analysis.
-   Keyword extraction.
-   General text summarization.
-   Medical-oriented summarization.
-   Interactive charts.

This application is separate from the ADK multi-agent workflow.

## 7. MCP Medication Information Server

The medication MCP server is configured in:

``` text
careinsight/medication_mcp_server.py
```

The Retrieval Agent starts it automatically using a standard
input/output connection.

The server calls the openFDA Drug Label API to retrieve available
medication information, including:

-   Indications and usage
-   Warnings
-   Adverse reactions
-   Drug interactions
-   Brand and generic medication names
-   Manufacturer information

You do not need to start the MCP server manually when using the ADK
workflow.

## 8. Run Automated Tests

Run the full test suite:

``` bash
uv run pytest tests/ -v
```

The latest recorded test execution completed with:

``` text
48 passed
```

Test coverage includes:

  Test file                   Purpose
  --------------------------- -------------------------------
  `test_critic.py`            Critic Agent validation
  `test_data_quality.py`      Dataset and schema checks
  `test_data_quality_ge.py`   Great Expectations validation
  `test_pipeline.py`          NLP pipeline behavior
  `test_sentiment.py`         Sentiment analysis
  `test_topics.py`            Topic modeling

The test suite checks application behavior and validation logic. Passing
tests do not establish clinical validity.

## 9. Understanding the Results

CareInsight AI produces different types of findings.

### Patient Review Analysis

Includes review counts, average ratings, sentiment distributions, and
topic groupings.

### FDA Medication Information

Contains information retrieved from official FDA drug labeling.

### Evidence Validation

The Critic Agent checks report consistency and evidence requirements.
Its output includes validation status and any detected errors or
warnings.

An approved validation result means the configured checks passed. It
should not be interpreted as proof that every generated statement is
medically correct.

## 10. Troubleshooting

### Missing Google API credentials

Check that `GOOGLE_API_KEY` is configured in the appropriate
environment.

For the ADK Gradio interface, check:

``` text
careinsight/.env
```

Do not paste the contents of this file into support requests.

### Missing Python packages

Confirm that the virtual environment is activated, then install the
requirements:

``` bash
uv pip install -r requirements.txt
```

### Dataset file not found

Verify that the expected CSV files are located in:

``` text
data/
```

### MCP or FDA retrieval errors

Check your internet connection and verify that the openFDA API is
reachable. The API may return no matching medication records or
encounter temporary service errors.

### Model download or loading errors

Some NLP models are downloaded from Hugging Face when first used. Verify
internet connectivity and allow sufficient disk space for model files.

### Port already in use

If Gradio cannot start because its default port is occupied, stop the
other process using that port or configure a different available port in
the relevant application's launch code.

## 11. Important Usage Disclaimer

CareInsight AI analyzes public, patient-reported medication reviews and
retrieves official medication labeling.

Patient experiences are subjective and should not be interpreted as
verified clinical findings. The application does not establish that a
medication caused any reported experience and must not be used to make
medication or treatment decisions.

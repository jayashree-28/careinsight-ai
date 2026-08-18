# careinsight-ai
AI-powered patient feedback analyser using NLP, MedGemma and LangChain
# 🏥 CareInsight AI — Patient Feedback Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)](https://gradio.app)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-yellow.svg)](https://huggingface.co)
[![Tests](https://img.shields.io/badge/Tests-29%20passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> AI-powered patient feedback analyser using NLP — sentiment analysis, 
> topic modelling, and side-by-side medical vs general AI summarisation.

---

## 🎯 What it does

CareInsight AI transforms unstructured patient reviews into actionable 
clinical insights using three NLP techniques:

| Feature | Model | What it produces |
|---|---|---|
| Sentiment Analysis | RoBERTa | POSITIVE / NEUTRAL / NEGATIVE + confidence |
| Topic Modelling | BERTopic | Automatically discovered clinical topics |
| General Summary | DistilBART | Plain-English summary |
| Medical Summary | Falconsai Medical | Clinically-aware summary |

---

## 🚀 Try it live

> **[🤗 Open in Hugging Face Spaces](https://huggingface.co/spaces/JayaBV/careinsight-ai)**

---

## 📊 Demo

### Tab 1 — Paste your own patient reviews
![Demo Tab 1](docs/demo_tab1.png)

### Tab 2 — Analyse by medical condition
![Demo Tab 2](docs/demo_tab2.png)

---

## 🏗️ Project structure

careinsight-ai/
├── src/
│ ├── sentiment.py # RoBERTa sentiment analysis
│ ├── topics.py # BERTopic topic modelling
│ ├── summariser.py # DistilBART general summarisation
│ ├── medgemma.py # Medical model summarisation
│ └── pipeline.py # LangChain orchestration
├── tests/
│ ├── test_sentiment.py # 7 sentiment tests
│ ├── test_topics.py # 7 topic tests
│ ├── test_pipeline.py # 6 pipeline tests
│ └── test_data_quality.py # 9 data quality tests
├── data/ # UCI Drug Review dataset
├── app.py # Gradio web application
└── requirements.txt # All dependencies


---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/YOURUSERNAME/careinsight-ai.git
cd careinsight-ai

# Create conda environment
conda create -n careinsight python=3.13 -y
conda activate careinsight

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
kaggle datasets download -d jessicali9530/kuc-hackathon-winter-2018 \
  -p data/ --unzip
```

---

## 🖥️ Run locally

```bash
conda activate careinsight
python3 app.py
```

Open browser at: `http://localhost:7860`

---

## 🧪 Run tests

```bash
python3 -m pytest tests/ -v
```


---

## 🤖 Models used

| Model | Size | Purpose |
|---|---|---|
| cardiffnlp/twitter-roberta-base-sentiment | 500MB | Sentiment classification |
| all-MiniLM-L6-v2 | 90MB | Sentence embeddings for BERTopic |
| sshleifer/distilbart-cnn-12-6 | 1.2GB | General summarisation |
| Falconsai/medical_summarization | 242MB | Medical summarisation |

---

## 📈 Dataset

**UCI Drug Review Dataset** — 161,297 real patient reviews
- Source: Kaggle
- Columns: drugName, condition, review, rating, date, usefulCount
- Conditions covered: Depression, Anxiety, Pain, Birth Control and more

---

## 🔬 Key findings — General vs Medical AI

| Condition | General Model | Medical Model |
|---|---|---|
| Depression | Surface-level mood summary | Specific drug names + side effect timeline |
| Anxiety | General improvement noted | Drug combinations (Paxil, Lexapro) + clinical terms |
| Pain | Patient quotes about relief | Injury history + medication escalation path |

---

## 🔮 Future enhancements

- [ ] MedGemma integration (pending HF approval)
- [ ] LangChain RAG for grounded clinical summaries
- [ ] Real-time sentiment dashboard
- [ ] Drift monitoring with Evidently AI
- [ ] Multilingual patient feedback support
- [ ] EHR API integration

---

## 🛠️ Tech stack

Python 3.13 — Core language
Hugging Face — Pre-trained AI models
BERTopic — Topic modelling
LangChain — Pipeline orchestration
Gradio — Web UI
Plotly — Interactive charts
pytest — Unit testing
Great Expectations — Data quality validation


---

## 👩‍💻 About

Built by an SDET transitioning into AI/ML Engineering.
This project demonstrates how testing expertise transfers 
directly into AI system validation.

---

## 📄 License

MIT License — free to use and extend.
# 🏦 Karen — AI Complaint Assistant
**MSBA Trends Project — Team 2**

> Turning millions of consumer banking complaints into clear, prioritized action items — accessible through a voice-enabled AI assistant.

> *This project repository is created in partial fulfillment of the requirements for the Big Data Analytics course offered by the Master of Science in Business Analytics program at the Carlson School of Management, University of Minnesota.*

---

## 👥 Team Members

Mohameddeq Ali · Cora Goodwin · Midori Neaton · Raja Sori · Xupei Ye · Kyle Zhu

---

## 📌 Project Overview

Financial institutions receive millions of consumer complaints. Manually reviewing them to identify what matters most — and what to fix first — is slow, resource-intensive, and prone to missing hidden risks buried in unstructured text.

**Karen** is an AI-powered complaint intelligence assistant that solves this in two steps:

1. **Structuring** — Raw narrative complaint text is transformed into distinct, analyzable categories using BERTopic topic modeling
2. **Prioritizing** — A weighted priority score ranks complaint categories by volume, growth, recency, topic danger, and narrative length

Decision makers can then talk to **Karen** in natural language — by voice or text — to query complaint metrics, generate visualizations, and surface ranked priorities instantly.

---

## 🎯 Business Value

| Pain Point | How Karen Helps |
|---|---|
| Data overload — millions of complaints impossible to review manually | BERTopic structures unstructured narratives into clear categories automatically |
| Hidden risks buried in complaint text | Priority scoring surfaces high-risk, fast-growing issues proactively |
| Slow manual analysis | Natural language queries replace hours of data mining |
| Inconsistent prioritization | Objective, weighted scoring ensures consistent, defensible decisions |

---

## 🗂️ Data Source

**CFPB Consumer Complaint Database (Public)**
- Link: [https://catalog.data.gov/dataset/consumer-complaint-database](https://catalog.data.gov/dataset/consumer-complaint-database)
- Size: ~14.35 million rows, ~8.4 GB
- Contains: structured complaint metadata + optional consumer narrative text
- ~26% of complaints include a consumer-written narrative (used for NLP)

> ⚠️ The full dataset is too large for GitHub. See setup instructions below for how to download it.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language & Runtime** | Python 3.11 (Miniconda/conda), WSL2 |
| **Web App** | Streamlit |
| **Data Storage & Querying** | DuckDB, Apache Parquet, Pandas |
| **Topic Modeling / NLP** | BERTopic (BERT-based), LDA |
| **LLM / AI** | OpenAI GPT-4o (agent, classification, TTS) |
| **Speech-to-Text** | Faster Whisper — large-v3-turbo model (CTranslate2 backend) |
| **Text-to-Speech** | OpenAI TTS API (tts-1, voice "nova") |
| **Agent Framework** | LangChain ReAct agent, PythonREPLTool, AgentExecutor |
| **Visualizations** | Plotly (rendered inside Streamlit) |
| **Audio** | Streamlit st.audio_input, Mutagen |
| **Environment Management** | python-dotenv |

---

## 🧠 How It Works

```
[CFPB Raw Complaints — 14.35M rows]
            │
            ▼
    [BERTopic NLP Pipeline]
    Narrative text → structured complaint categories
            │
            ▼
    [Priority Scoring Engine]
    Volume + Growth + Recency + Topic Danger + Length
            │
            ▼
    [Karen — AI Assistant]
    GPT-4o + Whisper + LangChain ReAct Agent
            │
            ▼
    [Streamlit Dashboard]
    Natural language Q&A · Voice input · Plotly charts
```

### Priority Score Formula

Each complaint category receives a weighted score combining:

- **Volume** — total complaint count in the category
- **Growth** — rate of increase over recent periods
- **Recency** — how recent the complaints are
- **Topic Danger** — risk signal derived from BERTopic cluster characteristics
- **Length** — narrative length as a proxy for complaint severity

### Karen's Capabilities

- 🎙️ **Voice input** — speak your question directly via browser microphone
- 💬 **Natural language queries** — ask in plain English, Karen writes and executes the analysis
- 📊 **On-demand visualizations** — Karen generates interactive Plotly charts on request
- 🔊 **Voice responses** — Karen can read her answers back to you via TTS
- 🏆 **Ranked priorities** — ask Karen which complaint categories need attention most urgently

---

## 📁 Project Structure

```
msba-team2-trendsproject/
│
├── app.py                    # Main Karen Streamlit application
├── file_manipulation.py      # Data processing and parquet handling
├── speach_to_text.ipynb      # Speech-to-text pipeline notebook
├── watch.py                  # File watcher / listener utility
├── inst.txt                  # Karen's system prompt and instructions
├── karen.png                 # Karen avatar / branding image
├── sample.m4a                # Sample audio file for testing voice input
├── MSBA_Market_Trends.py     # Original analysis script
│
├── data/                     # Data folder
│   └── (parquet files)       # Processed complaint data — not committed to GitHub
│
├── Karen Flyer.pdf           # Project overview handout
├── .gitignore
└── README.md                 # This file
```

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.11 (via [Miniconda](https://docs.conda.io/en/latest/miniconda.html) recommended)
- An OpenAI API key ([https://platform.openai.com](https://platform.openai.com))
- Git ([https://git-scm.com](https://git-scm.com))

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/ali00418-ship-it/msba-team2-trendsproject.git
cd msba-team2-trendsproject
```

---

### Step 2 — Create and activate a conda environment

```bash
conda create -n karen python=3.11 -y
conda activate karen
```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — Set up your API key

Create a file called `.env` in the project root and add your OpenAI key:

```
OPENAI_API_KEY=your-key-here
```

> ⚠️ Never commit your `.env` file — it is already listed in `.gitignore`

---

### Step 5 — Download the CFPB dataset

The full dataset (~8.4 GB) is too large for GitHub. Download it directly from:

👉 [https://catalog.data.gov/dataset/consumer-complaint-database](https://catalog.data.gov/dataset/consumer-complaint-database)

Place the downloaded CSV in the `data/` folder. The app expects parquet files — run `file_manipulation.py` to convert:

```bash
python file_manipulation.py
```

---

### Step 6 — Launch Karen

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to interact with Karen.

---


## 🔮 Future Use Cases

**Executive & Strategic**
- Monitor complaint trends across the entire portfolio in real time
- Identify fast-growing risk areas and allocate resources accordingly
- Query Karen directly for ranked priorities and visualizations — no manual reporting needed

**Front-Line Customer Service**
- Phone-based complaints are largely absent from the CFPB dataset — Karen closes this gap by transcribing call recordings and feeding them into the database
- Ensures phone complaint trends are captured and reflected in priority scores
- Helps representatives identify and resolve the highest-priority open complaints after calls, improving resolution speed and customer satisfaction

---

## ⚠️ Known Limitations

- Full BERTopic embedding on 3.7M narratives requires GPU (CPU runtime ~60+ hrs)
- `consumer_disputed` field is 94%+ null in recent data — not used in scoring
- Topic IDs from BERTopic are not stable across reruns — use topic keywords for reference

---

## 📚 References & Credits

- [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
- [OpenAI API](https://platform.openai.com/docs)
- [LangChain ReAct Agent](https://python.langchain.com/docs/modules/agents/)
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Streamlit](https://streamlit.io/)
- [DuckDB](https://duckdb.org/)

---

## 📬 Questions?

Open an [Issue](https://github.com/ali00418-ship-it/msba-team2-trendsproject/issues) on GitHub or reach out to any team member.

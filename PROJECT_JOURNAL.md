# 🏦 Banking Complaint Intelligence Dashboard — Project Journal

> **Team 2:** Mohameddeq Ali, Cora Goodwin, Midori Neaton, Raja Sori, Xupei Ye, Kyle Zhu  
> **Branch:** Personal dev branch (VSCode + Claude Code)  
> **Last updated:** 2026-04-16  
> **Purpose:** Running log of what has been done, decisions made, blockers, and next steps — so any future chat or teammate can pick up exactly where we left off.

---

## 📌 Project Summary (Quick Reference)

| Item | Detail |
|------|--------|
| **Goal** | Banking Complaint Intelligence Dashboard & Prioritization Engine |
| **Dataset** | CFPB Consumer Complaint Database (~5 GB CSV) |
| **Core tech** | Python, PySpark, BERTopic, HDBSCAN, SQL, Tableau |
| **AI role** | Narrative theme discovery (BERTopic) + optional LLM explanation layer |
| **Output** | Interactive dashboard with prioritized complaint themes |

---

## 📅 Session Log

### Session 1 — 2026-04-16
**Status:** Project kickoff / environment planning

#### ✅ What was done this session
- Read and reviewed the full project proposal (`Final_Proposal.docx`)
- Read and reviewed the course requirements (`Requirement.docx`)
- Researched what BERTopic is and how it works (see notes below)
- Planned strategy for handling the 5.02 GB CFPB dataset locally and in the pipeline
- Created the repo folder structure (`data/`, `notebooks/`, `src/`, `models/`, `outputs/`, `flier/`) plus `.gitignore`, `requirements.txt`, and `data/README_data.md`
- Started the first EDA notebook (`notebooks/01_eda_exploration.ipynb`): loads a 100k sample, saves it to `data/sample/cfpb_100k.csv`, counts total rows via chunked reading, and runs EDA on nulls, products/issues/companies, time trends, response/dispute/timeliness fields, and narrative length

#### 🧠 Key decisions made
- Will use a **sampled subset** of the data for local dev/exploration, and full data via PySpark for the real pipeline
- BERTopic will run only on rows **that have a non-null `consumer_complaint_narrative`** field
- Will pre-compute embeddings and save them separately so BERTopic doesn't re-embed every run

#### ❓ Open questions / blockers
- [ ] Does the team have access to cloud compute (Carlson IT)? Need to check.
- [ ] Which sentence-transformer model will we use? (`all-MiniLM-L6-v2` is fast and good for English)
- [ ] Where will the final dashboard live — Tableau Public or hosted?

#### 🔜 Next steps
- [x] Set up the GitHub repo with the required README.md structure
- [x] Create the project folder structure (see below)
- [ ] Load a sample of the CFPB data (see data strategy section) — *notebook written; needs to be run end-to-end*
- [ ] Run a first exploratory notebook on the sample — *notebook drafted; run it and review the charts*
- [ ] Install BERTopic and test it on a tiny slice of narratives

---

## 🗂️ Recommended Repo Folder Structure

```
banking-complaint-intelligence/
│
├── README.md                    # Project homepage (required)
├── PROJECT_JOURNAL.md           # This file — running log
│
├── data/
│   ├── sample/                  # Small sample CSV for dev (safe to commit)
│   └── README_data.md           # How to download the full dataset
│
├── notebooks/
│   ├── 01_exploration.ipynb     # EDA on sample data
│   ├── 02_bertopic_test.ipynb   # First BERTopic experiment
│   ├── 03_prioritization.ipynb  # Scoring/ranking logic
│   └── 04_pipeline_full.ipynb   # Full PySpark pipeline
│
├── src/
│   ├── ingest.py                # Data loading and cleaning
│   ├── bertopic_model.py        # BERTopic training and saving
│   ├── prioritization.py        # Scoring engine
│   └── utils.py                 # Helpers
│
├── models/
│   └── embeddings/              # Saved embeddings (gitignored if large)
│
├── outputs/
│   ├── topic_summary.csv        # Top themes output
│   └── priority_scores.csv      # Ranked complaint areas
│
├── flier/
│   └── team2_flier.pdf          # Event day handout (required)
│
├── .gitignore
└── requirements.txt
```

---

## 🤖 BERTopic — What It Is & How We'll Use It

### What is BERTopic?

BERTopic is a modern **topic modeling library** that finds hidden themes in a collection of text documents — automatically, without needing labeled data.

**It works in 4 steps:**

```
[Text narratives]
      ↓
1. EMBEDDING — each complaint narrative is converted to a dense vector
   using a pre-trained sentence transformer (e.g. all-MiniLM-L6-v2).
   This captures *meaning*, not just word frequency.
      ↓
2. DIMENSIONALITY REDUCTION — UMAP compresses those high-dimensional
   vectors into a 2D/5D space that preserves semantic structure.
      ↓
3. CLUSTERING — HDBSCAN groups similar complaints together into clusters.
   Complaints that don't fit any cluster become "outliers" (topic = -1).
      ↓
4. TOPIC LABELING — c-TF-IDF finds the most distinctive keywords for
   each cluster. That becomes the topic label (e.g. "overdraft_fee_account_charge").
```

### Why it's better than older methods (like LDA)
- LDA looks at word frequency — it doesn't understand *meaning*
- BERTopic understands that "unauthorized charge" and "fraudulent transaction" are about the same thing
- BERTopic also auto-detects the number of topics — you don't have to guess

### How we'll use it in this project

We will run BERTopic **inside** each CFPB Issue/Sub-issue category to find finer-grained themes:

```
CFPB Issue: "Incorrect information on your report"
  → BERTopic discovers sub-themes:
      - Theme A: "collection_agency_dispute_incorrect"
      - Theme B: "late_payment_removed_credit_bureau"
      - Theme C: "identity_theft_fraud_account"
```

This gives product teams much more actionable intelligence than just the broad Issue label.

### Key installation

```bash
pip install bertopic
pip install sentence-transformers
```

### Minimal working example (for our use case)

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load only rows with narratives
df = pd.read_csv("cfpb_sample.csv")
narratives = df[df["consumer_complaint_narrative"].notna()]["consumer_complaint_narrative"].tolist()

# Choose embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Train BERTopic
topic_model = BERTopic(embedding_model=embedding_model, language="english", verbose=True)
topics, probs = topic_model.fit_transform(narratives)

# See results
print(topic_model.get_topic_info())

# Save model for reuse (so we don't re-embed every time)
topic_model.save("models/bertopic_cfpb")
```

### Performance tips for 5GB+ data
- Pre-compute and **save embeddings as a numpy file** — embedding is the slowest step
- Run BERTopic on a **representative sample** (50k–200k narratives) first
- Use `min_topic_size=50` to avoid tiny irrelevant clusters
- Consider `nr_topics="auto"` to let it merge similar topics

```python
import numpy as np

# First run: compute and save embeddings
embeddings = embedding_model.encode(narratives, show_progress_bar=True)
np.save("models/cfpb_embeddings.npy", embeddings)

# Future runs: load pre-saved embeddings (instant!)
embeddings = np.load("models/cfpb_embeddings.npy")
topic_model.fit_transform(narratives, embeddings=embeddings)
```

---

## 💾 Data Strategy — How to Handle the 5.02 GB File

The CFPB CSV is too large to commit to GitHub and too slow to reload repeatedly. Here is the recommended approach:

### Option A — Local dev with a sample (recommended for you right now)

```python
import pandas as pd

# Load just 100,000 rows for exploration — takes seconds
df_sample = pd.read_csv(
    "complaints.csv",          # your local file path
    nrows=100_000
)
df_sample.to_csv("data/sample/cfpb_100k.csv", index=False)
```

Save that sample to `data/sample/cfpb_100k.csv` and **commit that** to GitHub (it'll be ~50–100 MB, manageable). Your teammates can use this sample for development too.

### Option B — Use chunked reading for processing the full file

```python
# Process in chunks without loading 5GB into RAM at once
chunks = pd.read_csv("complaints.csv", chunksize=50_000)
results = []
for chunk in chunks:
    # do filtering/cleaning per chunk
    filtered = chunk[chunk["consumer_complaint_narrative"].notna()]
    results.append(filtered)
df_narratives = pd.concat(results)
```

### Option C — PySpark for the full pipeline (course requirement)

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("CFPB_Complaints") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

df = spark.read.csv("complaints.csv", header=True, inferSchema=True)
df.createOrReplaceTempView("complaints")

# Example SQL query in Spark
result = spark.sql("""
    SELECT product, issue, COUNT(*) as complaint_count
    FROM complaints
    WHERE consumer_complaint_narrative IS NOT NULL
    GROUP BY product, issue
    ORDER BY complaint_count DESC
""")
result.show()
```

### What NOT to do
- ❌ Do NOT commit the full 5GB file to GitHub
- ❌ Do NOT `pd.read_csv("complaints.csv")` on the full file repeatedly — it's slow
- ✅ DO add `complaints.csv` and `*.npy` (embedding files) to `.gitignore`

### `.gitignore` entries to add

```
# Large data files
data/full/
*.csv.gz
complaints.csv

# Model artifacts
models/embeddings/*.npy
models/bertopic_cfpb/

# Python
__pycache__/
*.pyc
.env
```

---

## 🏗️ Architecture Overview

```
[CFPB Raw CSV 5GB]
        │
        ▼
[PySpark Ingestion & Cleaning]
        │
        ├──► [Structured Analysis]
        │         Product / Issue / Sub-issue
        │         Time trends, dispute rates
        │         Untimely response rates
        │
        └──► [Narrative Subset (has text)]
                  │
                  ▼
            [BERTopic Clustering]
                  │
                  ▼
            [Theme Discovery]
                  │
                  ▼
        [Prioritization Engine]
         volume + growth + dispute
         rate + anomaly detection
                  │
                  ▼
        [Tableau Dashboard]
         + Optional LLM summaries
```

---

## 📝 Notes & References

- BERTopic official docs: https://maartengr.github.io/BERTopic/
- CFPB dataset: https://www.consumerfinance.gov/data-research/consumer-complaints/
- Course requirement doc: `Requirement.docx`
- Project proposal: `Final_Proposal.docx`

---

*This file is a living document. Update it at the start and end of every work session.*

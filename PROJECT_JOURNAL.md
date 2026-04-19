# 🏦 Banking Complaint Intelligence Dashboard — Project Journal

> **Team 2:** Mohameddeq Ali, Cora Goodwin, Midori Neaton, Raja Sori, Xupei Ye, Kyle Zhu  
> **Branch:** Personal dev branch (VSCode + Claude Code)  
> **Last updated:** 2026-04-18  
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
- [x] Load a sample of the CFPB data (see data strategy section)
- [x] Run a first exploratory notebook on the sample
- [ ] Install BERTopic and test it on a tiny slice of narratives

---

### Session 2 — 2026-04-18
**Status:** Built the structured-signal baseline scorer (`notebooks/02_baseline.ipynb`)

#### ✅ What was done this session
- Built and executed `notebooks/02_baseline.ipynb` end-to-end on the 100k sample
- Set up the conda env (`msba`, Python 3.11) and installed all of `requirements.txt` plus `pyarrow`
- Made the first commit + push to the `kyle` branch on GitHub (`Set up project structure and first EDA notebook`)
- Wrote 7 outputs to `outputs/`: full + top-20 CSV, full Parquet, 3 PNG plots, plain-English `baseline_summary.md`
- Marked PySpark porting hot-spots in the notebook with `# TODO(pyspark):` comments

#### 🧠 Key decisions made
- **Unit of prioritization:** `product × issue` — 264 groups on the sample, median ~100 rows/group. `issue` alone loses product context; `product × issue × sub_issue` (605 groups) has too many tiny groups for stable rates on a 100k sample.
- **Growth-rate method:** last 12 months vs prior 12 months, with `+1` smoothing on denominator. Simpler than a regression slope and stable enough at this granularity.
- **Severity proxy:** use `monetary_relief_rate` instead of `consumer_disputed` (94.6% null in the sample).
- **Baseline weights (will be re-balanced once NLP signals land):** volume_z 0.30 · growth_z 0.35 · untimely 0.15 · monetary_relief 0.10 · recency 0.10
- **Z-score continuous signals only.** Rates already on [0,1] are used directly — easier for the team to interpret.

#### 📊 Top 5 baseline priorities (sample)
1. Credit reporting — Incorrect information on your report (35,209 complaints, +97% growth)
2. Student loan — Incorrect information on your report (small group, +400% growth — flagged as small-sample artifact)
3. Credit reporting — Improper use of your report (15,733 complaints, +43% growth)
4. Debt collection — Took or threatened to take negative or legal action (769, +320% growth, 5.3% untimely)
5. Payday loan — Struggling to pay your loan (small group, growth-driven)

#### ⚠️ Caveats / things to watch
- Smoothed growth still lets very small groups rank highly (rows 2 and 5 above). Consider a minimum-volume floor (e.g. require >= 50 prior-12-mo rows) when porting to the full file.
- Rankings on the full 14.35M-row file will likely differ — the 100k sample undersamples newer / regional issues.

#### 🔜 Next steps
- [ ] Notebook 03 — BERTopic theme discovery on rows with a narrative
- [ ] Notebook 04 — urgency / sentiment signals layered on top
- [ ] Re-balance weights once narrative signals land
- [ ] Port notebook 02 to PySpark for the full file (see `# TODO(pyspark):` markers)
- [ ] Hook `outputs/baseline_priorities_full.parquet` into Tableau

#### 🔁 Follow-up fix to Notebook 02 (same session, v1 → v2)
**Issue identified:** v1 of the scorer ranked 15/20 top groups with volume < 100 and 12/20 with volume < 50. Tiny groups with prior-window counts of 1–3 produced inflated growth rates (e.g. "2 → 10 complaints" = +400% growth) that dominated the score. Additionally, the v1 correlation heatmap showed `growth_rate ↔ recency_weight` = 0.62 — partial double-counting of "recent-ness."

**Fixes applied:**
- **Materiality floors on growth_z** — `MIN_TOTAL_VOLUME = 50`, `MIN_PRIOR_VOLUME = 20` on the 100k sample. Groups below either threshold get `growth_z_floored = 0`. Raw `growth_z` is kept in the output for transparency. Thresholds scale ~143× for the full file (≈ 500 / 200) — `# TODO(pyspark):` marked.
- **Weight rebalance:** untimely 0.15 → 0.20; recency 0.10 → 0.05. Volume / growth / monetary_relief unchanged. Rationale: recency was redundant with growth (corr 0.62); freed weight goes to untimely_rate (true regulatory-risk signal, low correlation to others).
- Added `prior_volume` as a column in the output so analysts can see why a group passed/failed the floor.
- Added a sanity-check cell in the notebook that asserts top-20 volume distribution.

**v2 sanity check (passed):**
- Top-20 groups with volume < 50: **2** (target ≤ 3) ✅
- Top-20 median volume: **353** (was much lower in v1)

**v2 top 5 priorities:**
1. Credit reporting — Incorrect information on your report (vol 35,209; +97% growth)
2. Credit reporting — Improper use of your report (vol 15,733; +43% growth)
3. Debt collection — Took or threatened to take negative or legal action (vol 769; +320% growth; 5.3% untimely)
4. Credit reporting — Problem with a company's investigation into an existing problem (vol 11,584; +59% growth)
5. Credit reporting, credit repair services, or other personal consumer reports — Incorrect information on your report (vol 7,184; floored growth — prior_volume = 0 likely a renamed/reclassified product category)

**Outstanding accepted limitation:** very small groups can still rank if they score highly on `monetary_relief_rate` or `untimely_rate` — this is intentional. A small group with 100% monetary relief is genuinely worth investigating; the floor only applies to growth.

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

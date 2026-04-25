"""
CFPB Complaint Intelligence Pipeline
=====================================
Steps:
  1. Load 9GB CSV in chunks, keep only useful columns
  2. Filter to TransUnion only + clean data
  3. Cluster with LDA (fast) and BERTopic (quality)
  4. Score each entry with a priority score (0-100)
  5. Output category labels + priority scores ready for visualization
"""

import pandas as pd
import numpy as np
import re
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────
CSV_PATH    = "/Users/mohameddeqali/Desktop/MSBA Trends Market/msba-team2-trendsproject/complaints.csv"
OUTPUT_PATH = "/Users/mohameddeqali/Desktop/MSBA Trends Market/msba-team2-trendsproject/complaints_labeled.csv"
CHUNK_SIZE          = 50_000
MIN_YEAR            = 2021
MIN_NARRATIVE_WORDS = 10
N_TOPICS_LDA        = 20
N_TOPICS_BERT       = 20
SAMPLE_FOR_BERT     = 100_000

# Priority score weights (must sum to 1.0)
WEIGHT_VOLUME   = 0.40   # how large is this topic cluster
WEIGHT_GROWTH   = 0.35   # how fast is this topic growing
WEIGHT_SEVERITY = 0.25   # how often does company fail to resolve

KEEP_COLS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company",
    "State",
    "Tags",
    "Consumer disputed?",
    "Company response to consumer",
]

# ─────────────────────────────────────────────
# STEP 1: LOAD IN CHUNKS + FILTER TO TRANSUNION
# ─────────────────────────────────────────────
print("📥 Loading CSV in chunks (TransUnion only)...")
chunks = []
for i, chunk in enumerate(pd.read_csv(
    CSV_PATH,
    usecols=KEEP_COLS,
    chunksize=CHUNK_SIZE,
    low_memory=False,
    on_bad_lines="skip"
)):
    # Filter to TransUnion early to keep memory down
    chunk = chunk[chunk["Company"].str.contains("TransUnion", case=False, na=False)]
    chunk["Date received"] = pd.to_datetime(chunk["Date received"], errors="coerce")
    chunk = chunk[chunk["Date received"].dt.year >= MIN_YEAR]
    chunk = chunk[chunk["Consumer complaint narrative"].notna()]
    chunks.append(chunk)
    if (i + 1) % 10 == 0:
        print(f"  ...processed {(i+1) * CHUNK_SIZE:,} raw rows")

df = pd.concat(chunks, ignore_index=True)
print(f"✅ Loaded {len(df):,} TransUnion rows after date + narrative filter\n")

# ─────────────────────────────────────────────
# STEP 2: CLEAN
# ─────────────────────────────────────────────
print("🧹 Cleaning data...")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"xx+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_narrative"] = df["Consumer complaint narrative"].apply(clean_text)
df["word_count"]      = df["clean_narrative"].apply(lambda x: len(x.split()))
df = df[df["word_count"] >= MIN_NARRATIVE_WORDS].reset_index(drop=True)
df["Product"] = df["Product"].str.strip().str.title()

print(f"✅ {len(df):,} rows after cleaning\n")

# ─────────────────────────────────────────────
# STEP 3A: LDA CLUSTERING
# ─────────────────────────────────────────────
print("🔵 Running LDA clustering...")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

vectorizer = CountVectorizer(
    max_df=0.90,
    min_df=10,
    max_features=5000,
    stop_words="english"
)
dtm = vectorizer.fit_transform(df["clean_narrative"])

lda = LatentDirichletAllocation(
    n_components=N_TOPICS_LDA,
    random_state=42,
    n_jobs=-1
)
lda_output = lda.fit_transform(dtm)

df["lda_topic"] = lda_output.argmax(axis=1)

feature_names = vectorizer.get_feature_names_out()
print("\n📋 LDA Topics (top 8 words each):")
for i, topic in enumerate(lda.components_):
    top_words = [feature_names[j] for j in topic.argsort()[:-9:-1]]
    print(f"  Topic {i:02d}: {', '.join(top_words)}")

# Human-readable topic labels based on actual LDA output
# Update these if you re-run LDA and topics shift
lda_topic_labels = {
    0:  "FCRA Reporting Compliance",
    1:  "Debt Collection Dispute",
    2:  "Identity Theft Block",
    3:  "Inaccurate Credit Info",
    4:  "Credit Bureau Investigation",
    5:  "FCRA Violation",
    6:  "Unauthorized Inquiry",
    7:  "TransUnion Direct Dispute",
    8:  "Reporting Item Removal",
    9:  "Late Payment Error",
    10: "FCRA Failure to Respond",
    11: "Inaccurate Consumer File",
    12: "Inaccuracy Investigation",
    13: "Unverified Debt Claim",
    14: "Inaccurate Account Balance",
    15: "Hard Inquiry / ID Theft",
    16: "Credit Score Impact",
    17: "Fair Credit Act Violation",
    18: "Unauthorized Data Use",
    19: "Fraudulent Account",
}

df["lda_topic_label"] = df["lda_topic"].map(lda_topic_labels).fillna("Other")
print("\n✅ LDA done\n")

# ─────────────────────────────────────────────
# STEP 3B: BERTOPIC CLUSTERING
# ─────────────────────────────────────────────
print(f"🟣 Running BERTopic on a sample of {SAMPLE_FOR_BERT:,} rows...")
print("   (This may take 10–30 mins on a laptop ☕)")

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    bert_df   = df.sample(n=min(SAMPLE_FOR_BERT, len(df)), random_state=42)
    docs      = bert_df["clean_narrative"].tolist()

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        nr_topics=N_TOPICS_BERT,
        verbose=True,
        calculate_probabilities=False,
        min_topic_size=20
    )
    topics, _ = topic_model.fit_transform(docs)

    bert_df = bert_df.copy()
    bert_df["bert_topic"] = topics
    # Filter out stop words from BERTopic labels for cleaner output
    BERT_STOP_WORDS = {
        "the", "and", "to", "of", "my", "is", "in", "it", "that", "this",
        "was", "for", "on", "are", "with", "they", "be", "at", "have",
        "from", "or", "an", "but", "not", "what", "all", "were", "when",
        "your", "has", "more", "do", "if", "about", "which", "their",
        "had", "would", "there", "can", "will", "one", "been", "these",
        "so", "up", "out", "she", "he", "said", "we", "you", "his",
        "her", "them", "no", "its", "also", "into", "than", "then",
        "by", "as", "any", "me", "am", "did", "our", "how", "get",
        "just", "ll", "ve", "re", "don", "doesn", "didn", "isn",
    }

    def clean_bert_label(t):
        if t == -1:
            return "Outlier"
        words = [w for w, _ in topic_model.get_topic(t)
                 if w.lower() not in BERT_STOP_WORDS and len(w) > 2]
        return " | ".join(words[:4]) if words else "Unlabeled"

    bert_df["bert_topic_label"] = bert_df["bert_topic"].apply(clean_bert_label)

    print("\n📋 BERTopic Topics:")
    topic_info = topic_model.get_topic_info()
    print(topic_info[["Topic", "Count", "Name"]].head(25).to_string(index=False))

    df = df.merge(
        bert_df[["bert_topic", "bert_topic_label"]],
        left_index=True, right_index=True,
        how="left"
    )
    topic_model.save("bertopic_model")
    print("\n✅ BERTopic done\n")

except ImportError:
    print("⚠️  BERTopic not installed. Skipping.\n")
    df["bert_topic"]       = np.nan
    df["bert_topic_label"] = np.nan

# ─────────────────────────────────────────────
# STEP 4: CATEGORY COLUMN
# Maps Product + LDA topic into a clean human-readable category
# ─────────────────────────────────────────────
print("🏷️  Assigning categories...")

# Map CFPB product names to clean category labels
product_category_map = {
    "Credit Reporting, Credit Repair Services, Or Other Personal Consumer Reports": "Credit Reporting",
    "Credit Reporting": "Credit Reporting",
    "Debt Collection": "Debt Collection",
    "Mortgage": "Mortgage",
    "Credit Card": "Credit Card",
    "Credit Card Or Prepaid Card": "Credit Card",
    "Student Loan": "Student Loan",
    "Bank Account Or Service": "Bank Account",
    "Checking Or Savings Account": "Bank Account",
    "Vehicle Loan Or Lease": "Auto Loan",
    "Payday Loan, Title Loan, Or Personal Loan": "Personal Loan",
    "Money Transfer, Virtual Currency, Or Money Service": "Money Transfer",
    "Prepaid Card": "Prepaid Card",
    "Other Financial Service": "Other",
}

df["category"] = df["Product"].map(product_category_map)

# Rescue rows still unmapped using the Issue column keywords
issue_category_map = {
    # Credit Reporting — core
    "incorrect information":                    "Credit Reporting",
    "inaccurate":                               "Credit Reporting",
    "credit reporting":                         "Credit Reporting",
    "investigation":                            "Credit Reporting",
    "dispute":                                  "Credit Reporting",
    "unable to get your credit report":         "Credit Reporting",
    "unable to get your credit score":          "Credit Reporting",
    "problem getting your report":              "Credit Reporting",
    "problem getting your free annual":         "Credit Reporting",
    "confusing or missing disclosures":         "Credit Reporting",
    "didn't provide services promised":         "Credit Reporting",
    # Unauthorized Inquiry / Improper Use — biggest bucket in Other
    "improper use of your report":              "Unauthorized Inquiry",
    "reporting company used your report":       "Unauthorized Inquiry",
    "credit inquiries on your report":          "Unauthorized Inquiry",
    "report provided to employer":              "Unauthorized Inquiry",
    "received unsolicited financial":           "Unauthorized Inquiry",
    "inquiry":                                  "Unauthorized Inquiry",
    "hard inquiry":                             "Unauthorized Inquiry",
    # Fraud / Identity Theft
    "fraudulent":                               "Fraud / Identity Theft",
    "identity theft":                           "Fraud / Identity Theft",
    "fraud":                                    "Fraud / Identity Theft",
    "unauthorized":                             "Fraud / Identity Theft",
    "trafficking":                              "Fraud / Identity Theft",
    # Debt Collection
    "debt":                                     "Debt Collection",
    "collection":                               "Debt Collection",
    # Loans / Other Products
    "loan":                                     "Loan Servicing",
    "mortgage":                                 "Mortgage",
    "billing":                                  "Credit Card",
    "getting a line of credit":                 "Credit Card",
    "problem when making payments":             "Bank Account",
    "account":                                  "Bank Account",
    "problem with customer service":            "Other - Customer Service",
    "confusing or misleading advertising":      "Other - Marketing",
    "charged upfront or unexpected fees":       "Other - Fees",
    "charged fees or interest":                 "Other - Fees",
}

# Also expand product map to catch remaining product variants
extra_product_map = {
    "Credit Reporting Or Other Personal Consumer Reports": "Credit Reporting",
    "Debt Or Credit Management":                          "Debt Collection",
    "Payday Loan, Title Loan, Personal Loan, Or Advance Loan": "Personal Loan",
}
df["category"] = df["category"].fillna(df["Product"].map(extra_product_map))

def map_by_issue(row):
    if pd.notna(row["category"]):
        return row["category"]
    issue = str(row.get("Issue", "")).lower()
    sub_issue = str(row.get("Sub-issue", "")).lower()
    # Check issue first, then sub-issue
    for keyword, cat in issue_category_map.items():
        if keyword in issue or keyword in sub_issue:
            return cat
    return "Other"

df["category"] = df.apply(map_by_issue, axis=1)

# Print rescue stats
total        = len(df)
other_count  = (df["category"] == "Other").sum()
rescued      = total - other_count
print(f"   Rescued {rescued:,} rows from 'Other' using Issue column")
print(f"   Remaining 'Other': {other_count:,}")
print(f"\nCategory breakdown:")
print(df["category"].value_counts().to_string())
print()
print(f"✅ Categories assigned\n")

# ─────────────────────────────────────────────
# STEP 5: PRIORITY SCORE PER ENTRY (0–100)
# Two-layer scoring:
#   TOPIC-LEVEL (shared across cluster):
#     - volume_score:   how large is this topic cluster (30%)
#     - growth_score:   MoM growth rate of this cluster (25%)
#     - severity_score: rate of unresolved responses in cluster (20%)
#   INDIVIDUAL-LEVEL (unique per complaint):
#     - disputed_score: consumer pushed back on resolution (10%)
#     - recency_score:  more recent = higher urgency (10%)
#     - length_score:   longer narrative = more detailed/severe (5%)
# ─────────────────────────────────────────────
print("🎯 Calculating priority scores...")

df["year_month"] = df["Date received"].dt.to_period("M").astype(str)
df["year"]       = df["Date received"].dt.year
df["month"]      = df["Date received"].dt.month
df["date_dt"]    = pd.to_datetime(df["Date received"])

# ── TOPIC-LEVEL SCORES ──

# Volume score: % of complaints per topic, normalized 0-100
topic_counts    = df["lda_topic"].value_counts()
topic_pct       = topic_counts / topic_counts.sum()
topic_vol_score = ((topic_pct - topic_pct.min()) /
                   (topic_pct.max() - topic_pct.min()) * 100)
df["volume_score"] = df["lda_topic"].map(topic_vol_score).fillna(0)

# Growth score: last 3 months vs prior 3 months per topic
max_date    = df["date_dt"].max()
recent_mask = df["date_dt"] >= (max_date - pd.DateOffset(months=3))
prior_mask  = (df["date_dt"] >= (max_date - pd.DateOffset(months=6))) & \
              (df["date_dt"] <  (max_date - pd.DateOffset(months=3)))

recent_counts = df[recent_mask]["lda_topic"].value_counts()
prior_counts  = df[prior_mask]["lda_topic"].value_counts()

growth_raw = {}
for t in df["lda_topic"].unique():
    r = recent_counts.get(t, 0)
    p = prior_counts.get(t, 1)
    growth_raw[t] = (r - p) / p

growth_series = pd.Series(growth_raw)
min_g, max_g  = growth_series.min(), growth_series.max()
growth_norm   = ((growth_series - min_g) / (max_g - min_g) * 100) if max_g != min_g else growth_series * 0 + 50
df["growth_score"] = df["lda_topic"].map(growth_norm).fillna(50)

# Severity score: unresolved rate per topic
bad_responses = ["Closed without relief", "In progress", "Untimely response"]
df["is_unresolved"] = df["Company response to consumer"].isin(bad_responses).astype(int)
topic_severity = df.groupby("lda_topic")["is_unresolved"].mean()
min_s, max_s   = topic_severity.min(), topic_severity.max()
severity_norm  = ((topic_severity - min_s) / (max_s - min_s) * 100) if max_s != min_s else topic_severity * 0 + 50
df["severity_score"] = df["lda_topic"].map(severity_norm).fillna(50)

# ── INDIVIDUAL-LEVEL SCORES ──

# Recency score: more recent = still happening = higher urgency (0-100)
min_date   = df["date_dt"].min()
date_range = (max_date - min_date).days
df["recency_score"] = ((df["date_dt"] - min_date).dt.days / date_range * 100).clip(0, 100)

# Length score: longer narrative = consumer still fighting = more serious (0-100, capped at 500 words)
df["length_score"] = (df["word_count"].clip(upper=500) / 500 * 100)

# Topic danger boost: inherently unresolved-by-nature topics score highest
# These topics represent systemic issues that rarely get truly resolved
HIGH_SEVERITY_TOPICS = {
    "Fraudulent Account",
    "Hard Inquiry / ID Theft",
    "Identity Theft Block",
    "FCRA Violation",
    "FCRA Failure to Respond",
    "Unverified Debt Claim",
    "Unauthorized Data Use",
    "Unauthorized Inquiry",
}
df["danger_boost"] = df["lda_topic_label"].apply(
    lambda t: 100 if t in HIGH_SEVERITY_TOPICS else 0
)

# ── FINAL COMBINED SCORE ──
# Focused on signals that capture ongoing/systemic unresolved pain
df["priority_score"] = (
    df["volume_score"]   * 0.25 +   # systemic = high volume
    df["growth_score"]   * 0.25 +   # getting worse = not resolved
    df["recency_score"]  * 0.15 +   # still happening now
    df["length_score"]   * 0.10 +   # consumer still fighting
    df["danger_boost"]   * 0.25     # inherently unresolved topic types
).round(1)

# Clip to 0-100
df["priority_score"] = df["priority_score"].clip(0, 100)

# Priority tier
def score_to_tier(s):
    if s >= 70: return "Critical"
    if s >= 50: return "High"
    if s >= 30: return "Medium"
    return "Low"

df["priority_tier"] = df["priority_score"].apply(score_to_tier)
print("✅ Priority scores calculated\n")
print(f"   Score range: {df['priority_score'].min()} – {df['priority_score'].max()}")
print(f"   Tier breakdown:")
print(df["priority_tier"].value_counts().to_string())
print()

# ─────────────────────────────────────────────
# STEP 6: SAVE
# ─────────────────────────────────────────────
print("💾 Saving labeled dataset...")

df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved → {OUTPUT_PATH}\n")

# ─────────────────────────────────────────────
# STEP 7: SUMMARY
# ─────────────────────────────────────────────
print("=" * 55)
print("📊 QUICK SUMMARY")
print("=" * 55)
print(f"\nTotal TransUnion complaints: {len(df):,}")
print(f"Date range: {df['Date received'].min().date()} → {df['Date received'].max().date()}")

print(f"\nTop 10 Categories:")
print(df["category"].value_counts().head(10).to_string())

print(f"\nPriority Tier Distribution:")
print(df["priority_tier"].value_counts().to_string())

print(f"\nTop 10 LDA Topics by Avg Priority Score:")
print(df.groupby("lda_topic_label")["priority_score"].mean().sort_values(ascending=False).head(10).round(1).to_string())

if "bert_topic_label" in df.columns and df["bert_topic_label"].notna().any():
    print(f"\nBERTopic Distribution (sampled rows):")
    print(df["bert_topic_label"].value_counts().head(10).to_string())

print("\n✅ Pipeline complete! Load complaints_labeled.csv into Tableau.")
import pandas as pd
import os
from datetime import datetime, date
import duckdb
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from openai import OpenAI
import json
import time

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("WHISP_KEY")

# --------------------------------------------------------------------------- #
# Issue → Priority Tier static mapping
# --------------------------------------------------------------------------- #
ISSUE_TIER_MAP = {
    "Advertising": "Medium",
    "Advertising and marketing, including promotional offers": "Medium",
    "Applying for a mortgage or refinancing an existing mortgage": "Medium",
    "Attempts to collect debt not owed": "Medium",
    "Charged fees or interest you didn't expect": "Low",
    "Charged upfront or unexpected fees": "Medium",
    "Closing an account": "Medium",
    "Closing on a mortgage": "Medium",
    "Closing your account": "Medium",
    "Communication tactics": "Low",
    "Confusing or misleading advertising or marketing": "Medium",
    "Confusing or missing disclosures": "Medium",
    "Credit limit changed": "Medium",
    "Credit monitoring or identity theft protection services": "Medium",
    "Dealing with your lender or servicer": "Medium",
    "Didn't provide services promised": "Medium",
    "Electronic communications": "Medium",
    "False statements or representation": "Medium",
    "Fees or interest": "Medium",
    "Fraud or scam": "Medium",
    "Getting a credit card": "Medium",
    "Getting a line of credit": "Low",
    "Getting a loan": "Low",
    "Getting a loan or lease": "Medium",
    "Getting the loan": "Low",
    "Identity theft protection or other monitoring services": "Medium",
    "Improper use of your report": "Medium",
    "Incorrect information on your report": "Medium",
    "Loan payment wasn't credited to your account": "Medium",
    "Managing an account": "Medium",
    "Managing the loan or lease": "Medium",
    "Opening an account": "Medium",
    "Other features, terms, or problems": "Medium",
    "Other transaction problem": "Medium",
    "Problem caused by your funds being low": "Medium",
    "Problem when making payments": "Low",
    "Problem with a company's investigation into an existing issue": "Medium",
    "Problem with a company's investigation into an existing problem": "Medium",
    "Problem with a credit reporting company's investigation into an existing problem": "Low",
    "Problem with a lender or other company charging your account": "Medium",
    "Problem with a purchase or transfer": "Medium",
    "Problem with a purchase shown on your statement": "Medium",
    "Problem with additional add-on products or services": "Medium",
    "Problem with customer service": "Medium",
    "Problem with fraud alerts or security freezes": "Medium",
    "Problem with the payoff process at the end of the loan": "Low",
    "Problems at the end of the loan or lease": "Low",
    "Received a loan you didn't apply for": "Medium",
    "Repossession": "Medium",
    "Struggling to pay mortgage": "Low",
    "Struggling to pay your bill": "Medium",
    "Struggling to pay your loan": "Medium",
    "Struggling to repay your loan": "Low",
    "Threatened to contact someone or share information improperly": "Medium",
    "Took or threatened to take negative or legal action": "Low",
    "Trouble during payment process": "Low",
    "Trouble using your card": "Medium",
    "Unable to get your credit report or credit score": "Medium",
    "Unauthorized transactions or other transaction problem": "Medium",
    "Unauthorized withdrawals or charges": "Medium",
    "Unexpected or other fees": "Medium",
    "Written notification about debt": "Low",
}

# The valid issue names are just the keys of the mapping
VALID_ISSUES = list(ISSUE_TIER_MAP.keys())


def get_priority_tier(issue):
    """Looks up the priority tier for a given issue. Falls back to 'Medium'."""
    return ISSUE_TIER_MAP.get(issue, "Medium")
def classify_issue(transcription):
    """
    Uses GPT-4o to classify a complaint transcription into one of the
    valid Issue categories. Returns the issue string.
    """
    client = OpenAI()

    issues_list = "\n".join(f"- {issue}" for issue in VALID_ISSUES)

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a complaint classification assistant. "
                    "Given a consumer complaint, classify it into exactly ONE of the "
                    "valid Issue categories listed below. Respond with ONLY the issue "
                    "category text, nothing else. No quotes, no explanation.\n\n"
                    f"Valid Issue categories:\n{issues_list}"
                ),
            },
            {
                "role": "user",
                "content": f"Classify this complaint:\n\n{transcription}",
            },
        ],
    )

    classified_issue = response.choices[0].message.content.strip()

    # Validate — if the LLM returned something not in the list, try fuzzy match
    if classified_issue not in VALID_ISSUES:
        # Case-insensitive match
        for valid in VALID_ISSUES:
            if valid.lower() == classified_issue.lower():
                classified_issue = valid
                break
        else:
            print(f"Warning: LLM returned unknown issue '{classified_issue}', defaulting to closest match.")
            # Fall back to first partial match
            for valid in VALID_ISSUES:
                if classified_issue.lower() in valid.lower() or valid.lower() in classified_issue.lower():
                    classified_issue = valid
                    break
            else:
                classified_issue = "Other service problem/ Irrelevent Complaint"

    return classified_issue


# --------------------------------------------------------------------------- #
# Core functions
# --------------------------------------------------------------------------- #

def clip(filename, n=1000):
    df = pd.read_csv(filename, n)
    df.to_csv('clipped.csv')


def covert_to_parquet(filename):
    csv_file = filename
    parquet_file = "data/data.parquet"
    print(f"Converting {csv_file} to {parquet_file}...")
    duckdb.sql(f"COPY (SELECT * FROM '{csv_file}') TO '{parquet_file}' (FORMAT PARQUET)")
    print("Conversion complete! You can now use data/data.parquet in your chatbot.")


def get_existing_files(directory_path):
    """Scans a directory and returns a set containing all current filenames."""
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return set()
    return {f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))}


def check_for_new_file(directory_path, existing_files_set):
    """
    Compares the current directory contents against a known set of filenames.
    Returns: (full_file_path, date_added) if a new file is found, otherwise None.
    """
    if not os.path.exists(directory_path):
        return None
    current_files = {f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))}
    new_files = current_files - existing_files_set
    if new_files:
        new_file_name = new_files.pop()
        full_path = os.path.join(directory_path, new_file_name)
        timestamp = os.path.getmtime(full_path)
        date_added = datetime.fromtimestamp(timestamp)
        return full_path, date_added
    return None


def append_transcription(transcription, parquet_dir, target_column):
    """
    Classifies a transcription, assigns a priority tier, then saves it as a
    parquet file in the labeled subfolder so DuckDB can query everything together.

    Pipeline:
      transcription → classify_issue() → get_priority_tier() → save to parquet
    """
    labeled_dir = os.path.join(parquet_dir, "labeled")
    os.makedirs(labeled_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(labeled_dir, f"recording_{timestamp}.parquet")
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Step 1: Classify the issue ---
    print("Classifying complaint issue...")
    issue = classify_issue(transcription)
    print(f"  → Issue: {issue}")

    # --- Step 2: Get priority tier ---
    priority_tier = get_priority_tier(issue)
    print(f"  → Priority Tier: {priority_tier}")

    # --- Step 3: Save as parquet ---
    safe_text = transcription.replace("'", "''")
    safe_issue = issue.replace("'", "''")

    duckdb.sql(f"""
        COPY (
            SELECT
                TIMESTAMP '{today}'   AS "Date received",
                '{safe_text}'         AS "{target_column}",
                '{safe_text}'         AS "clean_narrative",
                '{safe_issue}'        AS "Issue",
                '{priority_tier}'     AS "priority_tier",
                'Carlson Analytics Lab'             AS "Company",
                'MN'                  AS "State",
                CAST(NULL AS VARCHAR) AS "Product",
                CAST(NULL AS VARCHAR) AS "Sub-product",
                CAST(NULL AS VARCHAR) AS "Sub-issue",
                CAST(NULL AS VARCHAR) AS "Tags",
                CAST(NULL AS VARCHAR) AS "Company response to consumer",
                CAST(NULL AS VARCHAR) AS "Consumer disputed?",
                CAST(NULL AS VARCHAR) AS "lda_topic",
                CAST(NULL AS VARCHAR) AS "lda_topic_label",
                CAST(NULL AS VARCHAR) AS "bert_topic",
                CAST(NULL AS VARCHAR) AS "bert_topic_label",
                CAST(NULL AS VARCHAR) AS "category",
                CAST(NULL AS VARCHAR) AS "year_month",
                CAST(NULL AS INTEGER) AS "year",
                CAST(NULL AS INTEGER) AS "month",
                CAST(NULL AS DATE)    AS "date_dt",
                CAST(NULL AS DOUBLE)  AS "volume_score",
                CAST(NULL AS DOUBLE)  AS "growth_score",
                CAST(NULL AS INTEGER) AS "is_unresolved",
                CAST(NULL AS DOUBLE)  AS "severity_score",
                CAST(NULL AS DOUBLE)  AS "recency_score",
                CAST(NULL AS DOUBLE)  AS "length_score",
                CAST(NULL AS DOUBLE)  AS "danger_boost",
                CAST(NULL AS DOUBLE)  AS "priority_score",
                CAST(LENGTH('{safe_text}') - LENGTH(REPLACE('{safe_text}', ' ', '')) + 1 AS INTEGER) AS "word_count"
        ) TO '{out_path}' (FORMAT PARQUET)
    """)

    print(f"Transcription saved to '{out_path}' with Issue='{issue}', Tier='{priority_tier}'.\n")

    return issue, priority_tier


def transcribe(audio_path, model=None):
    """
    Transcribes an audio file.

    Parameters:
    - audio_path (str): Path to the audio file (e.g., .mp3, .wav).
    - model (WhisperModel, optional): A pre-loaded faster-whisper model object.
    """
    if model is None:
        print("Loading Whisper model into VRAM...")
        model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

    print(f"Transcribing '{os.path.basename(audio_path)}'...")
    segments, info = model.transcribe(audio_path, beam_size=5)

    transcription = " ".join([segment.text for segment in segments]).strip()
    print(f"Transcription complete. Length: {len(transcription)} characters.")

    return transcription
# 📂 Dataset — CFPB Consumer Complaint Database

## Source

**Consumer Financial Protection Bureau (CFPB) — Consumer Complaint Database (Public)**

- Official landing page: https://www.consumerfinance.gov/data-research/consumer-complaints/
- Direct download (data.gov): https://catalog.data.gov/dataset/consumer-complaint-database
- Full CSV download: https://files.consumerfinance.gov/ccdb/complaints.csv.zip

The dataset contains consumer complaints submitted to the CFPB about banks, credit cards, debt collectors, mortgages, student loans, and other financial products. It is updated daily by the CFPB.

## Size & format

- Full file: ~8.4 GB uncompressed CSV (`complaints.csv`)
- Rows: ~5 million+ (grows over time)
- Encoding: UTF-8

## How to get the full dataset

1. Visit https://www.consumerfinance.gov/data-research/consumer-complaints/
2. Click **"Download the data"**
3. Choose the **CSV** format
4. Unzip and place the file at the repo root as `complaints.csv`

> ⚠️ The full CSV is in `.gitignore` and must **not** be committed to GitHub.

## What's in the repo

We commit only a small sample of the data so teammates can run notebooks without downloading the full file:

- `data/sample/cfpb_100k.csv` — first 100,000 rows of the full dataset, created by `notebooks/01_eda_exploration.ipynb`

For the full pipeline, use PySpark on the local full file (see `PROJECT_JOURNAL.md` → Data Strategy).

## Key columns (common CFPB schema)

| Column (normalized) | Description |
|---|---|
| `date_received` | Date complaint was received |
| `product` / `sub_product` | Financial product the complaint is about |
| `issue` / `sub_issue` | Type of problem reported |
| `consumer_complaint_narrative` | Free-text narrative (often null — only present when consumer opts in) |
| `company` | Company the complaint is filed against |
| `state` / `zip_code` | Consumer location |
| `company_response_to_consumer` | How the company responded |
| `timely_response` | Yes / No — was the company's response timely |
| `consumer_disputed` | Yes / No — did the consumer dispute the resolution |
| `complaint_id` | Unique ID |

Note: Raw CFPB column names contain spaces and question marks (e.g. `"Consumer complaint narrative"`, `"Timely response?"`). Our notebooks normalize them to snake_case at load time.

# Karen: Voice-to-Insights Analytics Agent

>A consumer complaint analytics system that combines topic modeling, priority scoring, a voice transcription pipeline, and an intelligent data chatbot. Karen processes raw CFPB complaint data through an NLP labeling pipeline, ingests new audio recordings in real time, automatically transcribes and classifies them, assigns a priority tier, stores them as structured data, and lets users query everything through a conversational interface.

> *This project repository is created in partial fulfillment of the requirements for the Big Data Analytics course offered by the Master of Science in Business Analytics program at the Carlson School of Management, University of Minnesota.*
---

## Team Members

Mohameddeq Ali · Cora Goodwin · Midori Neaton · Raja Sori · Xupei Ye · Kyle Zhu

---

## Project Overview

Financial institutions receive millions of consumer complaints. Manually reviewing them to identify what matters most and what to fix first is slow, resource-intensive, and prone to missing hidden risks buried in unstructured text.

**Karen** is an AI-powered complaint intelligence assistant that solves this in two steps:

1. **Structuring**: Raw narrative complaint text is transformed into distinct, analyzable categories using BERTopic topic modeling
2. **Prioritizing**: A weighted priority score ranks complaint categories by volume, growth, recency, topic danger, and narrative length

Decision makers can then talk to **Karen** in natural language, either by voice or text, to query complaint metrics, generate visualizations, and surface ranked priorities instantly.

---

## Business Value

| Pain Point | How Karen Helps |
|---|---|
| Data overload — millions of complaints impossible to review manually | BERTopic structures unstructured narratives into clear categories automatically |
| Hidden risks buried in complaint text | Priority scoring surfaces high-risk, fast-growing issues proactively |
| Slow manual analysis | Natural language queries replace hours of data mining |
| Inconsistent prioritization | Objective, weighted scoring ensures consistent, defensible decisions |

---

##  Data Source

**CFPB Consumer Complaint Database (Public)**
- Link: [https://catalog.data.gov/dataset/consumer-complaint-database](https://catalog.data.gov/dataset/consumer-complaint-database)
- Size: ~14.35 million rows, ~8.4 GB
- Contains: structured complaint metadata + optional consumer narrative text
- ~26% of complaints include a consumer-written narrative (used for NLP)

> ⚠️ The full dataset is too large for GitHub. See setup instructions below for how to download it.

---

### Priority Score Formula

Each complaint category receives a weighted score combining:

- **Volume**: total complaint count in the category
- **Growth**: rate of increase over recent periods
- **Recency**: how recent the complaints are
- **Topic Danger**: risk signal derived from BERTopic cluster characteristics
- **Length**: narrative length as a proxy for complaint severity

### Karen's Capabilities

- **Voice input**: speak your question directly via browser microphone
- **Natural language queries**: ask in plain English, Karen writes and executes the analysis
- **On-demand visualizations**: Karen generates interactive Plotly charts on request
- **Voice responses**: Karen can read her answers back to you via TTS
- **Ranked priorities**: ask Karen which complaint categories need attention most urgently

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Environment Setup](#environment-setup)
5. [API Keys](#api-keys)
6. [Installation](#installation)
7. [Data Setup](#data-setup)
8. [Running the Modeling and Scoring Pipeline](#running-the-modeling-and-scoring-pipeline)
9. [Running the Application](#running-the-application)
10. [Usage](#usage)
11. [GPU vs CPU Configuration](#gpu-vs-cpu-configuration)
12. [Project Structure](#project-structure)
13. [Tools and Technologies](#tools-and-technologies)
14. [Troubleshooting](#troubleshooting)

---

## Features

- **NLP labeling pipeline**: LDA and BERTopic topic modeling with composite priority scoring on raw CFPB data
- **Automatic transcription**: New audio recordings in a watched folder are transcribed using OpenAI Whisper
- **Complaint classification**: Each transcription is classified into one of 62 Issue categories using GPT-4o
- **Priority assignment**: Each complaint is assigned a priority tier (Critical, High, Medium, Low) via a static mapping
- **Structured storage**: Classified complaints are stored as individual Parquet files queryable by DuckDB
- **Conversational analytics**: Ask questions in plain English and get answers, tables, or interactive Plotly charts
- **Optional voice I/O**: Speak your questions and hear Karen's answers read aloud

---

## Architecture

The system has four components. The modeling pipeline runs once to produce the labeled dataset. The other three run together in production.

```
[complaints.csv]
    |
    v
[modeling_scoring.py] -- LDA + BERTopic + priority scoring --> complaints_labeled.csv
    |
    v
[file_manipulation.py] -- covert_to_parquet() --> data/labeled/complaints_labeled.parquet
    |
    v
[data/labeled/*.parquet] <-- structured complaint data
    ^                  ^
    |                  |
[app.py]          [watch.py + file_manipulation.py]
 Streamlit           Audio --> Whisper --> GPT-4o classify
 chatbot             --> assign tier --> save Parquet
```

1. **modeling_scoring.py**: One-time NLP pipeline that loads raw CFPB CSV data, runs LDA and BERTopic topic modeling, assigns categories, calculates priority scores (0-100), and outputs a labeled CSV
2. **file_manipulation.py**: Utility functions for CSV-to-Parquet conversion, audio transcription, GPT-4o issue classification, priority tier lookup, and Parquet storage
3. **watch.py**: Background watcher that detects new audio files in a directory and triggers the transcription/classification pipeline
4. **app.py**: Streamlit chatbot with LangChain ReAct agent, optional voice input/output

---

## Prerequisites

- **Operating System**: Windows 10/11 with WSL2, or native Linux/macOS
- **Python**: 3.10 or 3.11 (3.11 recommended)
- **GPU (optional but recommended)**: NVIDIA GPU with CUDA support for fast Whisper transcription and BERTopic embedding
- **CPU alternative**: Whisper runs on CPU with INT8 quantization (slower but fully functional). BERTopic will also run on CPU but significantly slower.
- **Internet access**: Required for OpenAI API calls (GPT-4o, TTS)
- **Raw data**: The CFPB complaints CSV file (downloadable from https://www.consumerfinance.gov/data-research/consumer-complaints/)

---

## Environment Setup

### Step 1: Install WSL2 (Windows only)

If you are on Windows, you need WSL2 to run the project. Open PowerShell as Administrator and run:

```powershell
wsl --install
```

This installs Ubuntu by default. Restart your computer when prompted. After restart, open Ubuntu from the Start menu and create a username and password.

Update the system packages:

```bash
sudo apt update && sudo apt upgrade -y
```

### Step 2: Install Miniconda

Download and install Miniconda inside WSL2 (or on your Linux/macOS machine):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the prompts and say "yes" to initializing conda. Then close and reopen your terminal, or run:

```bash
source ~/.bashrc
```

### Step 3: Create the Conda Environment

```bash
conda create -n chatbot_env python=3.11 -y
conda activate chatbot_env
```

You should see `(chatbot_env)` at the beginning of your terminal prompt.

### Step 4: Install CUDA Toolkit (GPU users only)

If you have an NVIDIA GPU and want fast transcription and embedding, install the CUDA toolkit inside your conda environment:

```bash
conda install -c conda-forge cudatoolkit=12.1 -y
```

If you do NOT have an NVIDIA GPU, skip this step. See the [GPU vs CPU Configuration](#gpu-vs-cpu-configuration) section for how to configure Whisper for CPU.

---

## API Keys

This project requires API keys from two services.

### OpenAI API Key

Used for: GPT-4o (chatbot agent + complaint classification) and TTS (voice responses)

1. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (it starts with `sk-`)

### Hugging Face Token

Used for: Downloading the Whisper model weights and the sentence-transformers model for BERTopic

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Sign in or create an account
3. Click "New token", give it a name, select "Read" access
4. Copy the token (it starts with `hf_`)

### Create the .env File

In the project root directory, create a file called `.env`:

```bash
touch .env
```

Open it in a text editor and add your keys:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
WHISP_KEY=hf_your-hugging-face-token-here
```

Do NOT add quotes around the values. Do NOT commit this file to git.

Make sure `.env` is in your `.gitignore`:

```bash
echo ".env" >> .gitignore
```

---

## Installation

With your conda environment activated (`conda activate chatbot_env`), install all required packages.

### Core Dependencies (chatbot + voice pipeline)

```bash
pip install --break-system-packages \
    streamlit \
    duckdb \
    pandas \
    numpy \
    plotly \
    python-dotenv \
    openai \
    faster-whisper \
    langchain \
    langchain-openai \
    langchain-experimental \
    langchain-classic \
    mutagen \
    fastparquet
```

### Modeling and Scoring Dependencies

These are required to run `modeling_scoring.py`:

```bash
pip install --break-system-packages \
    scikit-learn \
    bertopic \
    sentence-transformers
```

Note: `sentence-transformers` downloads the `all-MiniLM-L6-v2` embedding model on first run (about 80MB).

### GPU-Specific Install (NVIDIA GPU users)

If you have an NVIDIA GPU and installed the CUDA toolkit in Step 4:

```bash
pip install ctranslate2
```

This should automatically pick up CUDA. Verify with:

```bash
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
```

If it prints a number greater than 0, CUDA is working.

### CPU-Only Install (no NVIDIA GPU)

If you do NOT have an NVIDIA GPU:

```bash
pip install ctranslate2
```

Then see the [GPU vs CPU Configuration](#gpu-vs-cpu-configuration) section to configure the code for CPU mode.

---

## Data Setup

### Step 1: Download the Raw CFPB Data

Download the consumer complaints CSV from the CFPB website:

https://www.consumerfinance.gov/data-research/consumer-complaints/

Save the file as `complaints.csv` in the project root directory.

### Step 2: Create the Data Directories

```bash
mkdir -p data/labeled
```

### Step 3: Set Up the Audio Recordings Directory

The file watcher monitors a directory for new audio recordings. By default this is set to:

```
/mnt/c/Users/Raja/Documents/Sound Recordings
```

To change this, edit the `PATH` variable in `watch.py`:

```python
PATH = "/path/to/your/audio/recordings"
```

On WSL2, Windows paths are accessible under `/mnt/c/`. For example, `C:\Users\YourName\Documents` becomes `/mnt/c/Users/YourName/Documents`.

### Step 4: Add the Karen Avatar

Place an image file called `karen.png` in the project root directory. This is used as the chatbot avatar in the Streamlit UI.

---

## Running the Modeling and Scoring Pipeline

This pipeline runs once to transform the raw CFPB CSV into a labeled, scored dataset. You do not need to re-run it unless you want to refresh the data or change model parameters.

### Step 1: Configure the Script

Open `modeling_scoring.py` and update the file paths at the top:

```python
CSV_PATH    = "complaints.csv"          # path to your raw CFPB CSV
OUTPUT_PATH = "complaints_labeled.csv"  # where to save the labeled output
```

You can also adjust these parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 50,000 | Number of CSV rows to process per chunk (adjust based on your RAM) |
| `MIN_YEAR` | 2021 | Only keep complaints from this year onward |
| `MIN_NARRATIVE_WORDS` | 10 | Discard complaints with fewer words in the narrative |
| `N_TOPICS_LDA` | 20 | Number of LDA topics to extract |
| `N_TOPICS_BERT` | 20 | Number of BERTopic topics to extract |
| `SAMPLE_FOR_BERT` | 100,000 | Number of rows to sample for BERTopic (reduces runtime) |

### Step 2: Run the Pipeline

```bash
conda activate chatbot_env
python modeling_scoring.py
```

The pipeline runs through these stages:

1. **Load and filter**: Reads the CSV in chunks, filters to the target company and date range, keeps only rows with complaint narratives
2. **Clean text**: Lowercases, removes masked characters (XX), strips non-alphabetic characters, filters by word count
3. **LDA topic modeling**: Fits a Latent Dirichlet Allocation model using scikit-learn's CountVectorizer and assigns each complaint a topic number and human-readable label
4. **BERTopic modeling**: Runs transformer-based topic modeling using the `all-MiniLM-L6-v2` sentence embedding model. This step samples a subset of rows for performance and can take 10-30 minutes on a laptop. If BERTopic is not installed, this step is skipped gracefully.
5. **Category assignment**: Maps each complaint to a human-readable category using the Product column, LDA topic labels, and Issue column keywords as fallbacks
6. **Priority scoring**: Calculates a composite score (0-100) per complaint using six weighted signals:
   - Volume score (25%): How large is the complaint's topic cluster
   - Growth score (25%): Month-over-month growth rate of the cluster
   - Recency score (15%): How recent the complaint is
   - Length score (10%): Longer narratives indicate consumers still actively fighting
   - Danger boost (25%): Systemic high-severity topics (fraud, identity theft, FCRA violations) get a full boost
   - These combine into a priority_score which maps to a tier: Critical (70+), High (50-69), Medium (30-49), Low (0-29)
7. **Save**: Outputs `complaints_labeled.csv`

Expected output:

```
Loading CSV in chunks (TransUnion only)...
Loaded X,XXX TransUnion rows after date + narrative filter
Cleaning data...
Running LDA clustering...
Running BERTopic...
Assigning categories...
Calculating priority scores...
Saving labeled dataset...
QUICK SUMMARY
```

### Step 3: Convert the Labeled CSV to Parquet

The chatbot reads Parquet files, not CSVs. Use the built-in conversion function in `file_manipulation.py`.

First, update the output path in the `covert_to_parquet` function in `file_manipulation.py` if needed. By default it writes to `data/data.parquet`. You can either update it or move the file after conversion.

Run the conversion:

```bash
python -c "import file_manipulation as fm; fm.covert_to_parquet('complaints_labeled.csv')"
```

This uses DuckDB to stream the CSV into Parquet format without loading the entire file into memory.

Move the output into the labeled directory:

```bash
mv data/data.parquet data/labeled/complaints_labeled.parquet
```

Verify the conversion worked:

```bash
python -c "
import duckdb
print(duckdb.sql(\"SELECT COUNT(*) AS rows FROM 'data/labeled/complaints_labeled.parquet'\").fetchdf())
"
```

---

## Running the Application

The application has two components that run simultaneously in separate terminals.

### Terminal 1: Start the File Watcher

```bash
conda activate chatbot_env
python watch.py
```

This will start polling for new audio files. You should see:

```
No new file found
No new file found
...
```

When a new recording appears, it will automatically:
1. Wait for the file to finish saving
2. Transcribe the audio with Whisper
3. Classify the Issue using GPT-4o
4. Look up the priority tier
5. Save the result as a new Parquet file in `data/labeled/`

### Terminal 2: Start the Chatbot

```bash
conda activate chatbot_env
streamlit run app.py
```

This opens the Streamlit UI in your browser (usually at `http://localhost:8501`).

---

## Usage

### Text Chat

Type your question in the chat input at the bottom of the page. Examples:

- "How many complaints are in the dataset?"
- "What are the top 10 companies by complaint count?"
- "Show me a bar chart of complaints by priority tier"
- "What is the latest complaint? Give me the company, date, and full narrative."
- "Compare fraud complaints in California vs Texas"
- "What LDA topics have the highest average priority score?"
- "Show me the distribution of priority tiers as a pie chart"
- "How many Critical priority complaints were filed this year?"

### Voice Input

1. Toggle on "Voice input" in the sidebar
2. Click the microphone widget and speak your question
3. Wait for transcription to complete
4. Karen processes your question and responds

### Voice Output

1. Toggle on "Voice responses" in the sidebar
2. Karen will read her answers aloud using OpenAI's TTS

### Recording New Complaints

1. Make sure `watch.py` is running in a separate terminal
2. Record an audio file and save it to your watched directory
3. The watcher will detect it within 5-10 seconds
4. After processing, the complaint appears in Karen's data immediately

---

## GPU vs CPU Configuration

### GPU Configuration (default)

The project is configured for NVIDIA GPU by default. The relevant settings are in two files:

**file_manipulation.py** (transcribe function):
```python
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
```

**app.py** (load_whisper function):
```python
return WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
```

### CPU Configuration

If you do not have an NVIDIA GPU, you need to change both files.

**file_manipulation.py**: Find the `transcribe` function and change the model loading line:

```python
# Change this:
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

# To this:
model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
```

**app.py**: Find the `load_whisper` function and change it:

```python
# Change this:
return WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

# To this:
return WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
```

**Important notes for CPU users:**

- Transcription will be significantly slower (roughly 5-10x depending on your CPU)
- The `int8` compute type reduces memory usage and improves CPU performance
- BERTopic in `modeling_scoring.py` will also run slower on CPU but requires no code changes
- If you still get CUDA-related errors on import, add this at the very top of both `file_manipulation.py` and `app.py`, before any other imports:

```python
import os
os.environ["CT2_USE_CUDA"] = "0"
```

- You can also use a smaller model for faster CPU performance at the cost of accuracy:

```python
model = WhisperModel("base", device="cpu", compute_type="int8")
```

Available model sizes from smallest to largest: `tiny`, `base`, `small`, `medium`, `large-v3-turbo`, `large-v3`

---

## Project Structure

```
project-root/
|
|-- app.py                    # Streamlit chatbot application
|-- watch.py                  # Background file watcher for new recordings
|-- file_manipulation.py      # Transcription, classification, storage, CSV-to-Parquet
|-- modeling_scoring.py       # NLP pipeline: LDA, BERTopic, category, priority scoring
|-- karen.png                 # Chatbot avatar image
|-- complaints.csv            # Raw CFPB data (not committed to git)
|-- complaints_labeled.csv    # Output of modeling_scoring.py (not committed to git)
|-- .env                      # API keys (not committed to git)
|-- .gitignore
|
|-- data/
|   |-- labeled/
|   |   |-- complaints_labeled.parquet         # Converted from complaints_labeled.csv
|   |   |-- recording_20260423_143207.parquet   # Auto-generated from audio
|   |   |-- recording_20260423_151022.parquet   # Auto-generated from audio
|   |   |-- ...
```

---

## Tools and Technologies

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11 | Runtime language |
| Faster Whisper | latest | Speech-to-text transcription (CTranslate2 backend) |
| OpenAI GPT-4o | API | LLM for data analysis agent, complaint classification |
| OpenAI TTS (tts-1) | API | Text-to-speech for voice responses |
| LangChain | latest | ReAct agent framework with PythonREPLTool |
| DuckDB | latest | SQL engine that queries Parquet files directly, CSV-to-Parquet conversion |
| Apache Parquet | -- | Columnar data storage format |
| Streamlit | latest | Web application framework for the chat UI |
| Plotly | latest | Interactive chart generation |
| Pandas | latest | Data manipulation and display formatting |
| NumPy | latest | Numerical operations in the scoring pipeline |
| scikit-learn | latest | CountVectorizer and LDA topic modeling |
| BERTopic | latest | Transformer-based topic modeling |
| sentence-transformers | latest | all-MiniLM-L6-v2 embedding model for BERTopic |
| Mutagen | latest | MP3 duration detection for voice playback timing |
| Miniconda | latest | Python environment management |
| WSL2 | -- | Linux runtime on Windows with GPU passthrough |

---

## Troubleshooting

### "Library libcublas.so.12 is not found or cannot be loaded"

You are running on CPU but CTranslate2 is trying to load CUDA libraries. Add this to the very top of your Python files, before any other imports:

```python
import os
os.environ["CT2_USE_CUDA"] = "0"
```

Also make sure you changed `device="cuda"` to `device="cpu"` in both `file_manipulation.py` and `app.py`.

### "Table Function with name clear_cache does not exist"

Your version of DuckDB does not support `clear_cache()`. This is handled automatically in the latest version of the code. The agent creates a fresh `duckdb.connect()` for each query instead of relying on cache clearing.

### KeyError in fastparquet time_factors

This happens when appending to Parquet files with INT96 timestamp encoding. The project avoids this entirely by writing individual Parquet files per recording via DuckDB instead of appending to a single file.

### Chatbot does not see newly recorded complaints

Make sure the agent prompt instructs GPT-4o to use `con = duckdb.connect()` instead of `duckdb.sql()`. The default connection caches glob results and will not pick up new files written by `watch.py` during the same session.

### modeling_scoring.py runs out of memory

Reduce `CHUNK_SIZE` (e.g., from 50,000 to 20,000) and `SAMPLE_FOR_BERT` (e.g., from 100,000 to 50,000) at the top of the script. BERTopic is the most memory-intensive step.

### BERTopic is not installed or import fails

The pipeline handles this gracefully. If BERTopic cannot be imported, the script skips that step and fills the `bert_topic` and `bert_topic_label` columns with NaN. The rest of the pipeline (LDA, categories, priority scores) still runs normally.

### CSV-to-Parquet conversion fails

Make sure DuckDB is installed and the CSV file path is correct. If you get memory errors on very large CSVs, DuckDB's COPY command streams data and should not run out of memory. Check that the output directory exists:

```bash
mkdir -p data/labeled
```

### Voice input disappears after first use

This is handled by the dynamic key pattern in `app.py`. The `st.audio_input` widget key increments after each recording, forcing Streamlit to render a fresh widget. If you modified `app.py`, make sure the `voice_key_counter` logic and `st.rerun()` calls are intact.

### Voice response gets cut off

The code uses `mutagen` to detect MP3 duration and waits that long before triggering `st.rerun()`. Make sure `mutagen` is installed:

```bash
pip install mutagen
```

If responses are still clipped, increase the buffer time in `app.py` by changing `time.sleep(duration + 0.25)` to `time.sleep(duration + 1.0)`.

### Streamlit shows "ModuleNotFoundError"

Make sure your conda environment is activated before running:

```bash
conda activate chatbot_env
streamlit run app.py
```

### OpenAI API errors (401 Unauthorized)

Your API key is missing or invalid. Check that your `.env` file exists in the project root and contains a valid key:

```env
OPENAI_API_KEY=sk-your-key-here
```

Make sure there are no extra spaces or quotes around the value.

---
## Questions?

Open an [Issue](https://github.com/ali00418-ship-it/msba-team2-trendsproject/issues) on GitHub or reach out to any team member.

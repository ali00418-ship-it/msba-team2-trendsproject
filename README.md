# Karen: Voice-to-Insights Analytics Agent

A consumer complaint analytics system that combines a voice transcription pipeline with an intelligent data chatbot. Karen ingests audio recordings, automatically transcribes and classifies them, assigns a priority tier, stores them as structured data, and lets users query everything through a conversational interface.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Environment Setup](#environment-setup)
5. [API Keys](#api-keys)
6. [Installation](#installation)
7. [Data Setup](#data-setup)
8. [Running the Project](#running-the-project)
9. [Usage](#usage)
10. [GPU vs CPU Configuration](#gpu-vs-cpu-configuration)
11. [Project Structure](#project-structure)
12. [Tools and Technologies](#tools-and-technologies)
13. [Troubleshooting](#troubleshooting)

---

## Features

- **Automatic transcription**: New audio recordings in a watched folder are transcribed using OpenAI Whisper
- **Complaint classification**: Each transcription is classified into one of 62 Issue categories using GPT-4o
- **Priority assignment**: Each complaint is assigned a priority tier (Critical, High, Medium, Low) via a static mapping
- **Structured storage**: Classified complaints are stored as individual Parquet files queryable by DuckDB
- **Conversational analytics**: Ask questions in plain English and get answers, tables, or interactive Plotly charts
- **Optional voice I/O**: Speak your questions and hear Karen's answers read aloud

---

## Architecture

The system has three independent components that communicate through the filesystem:

```
Audio File
    |
    v
[watch.py] -- polls directory for new recordings
    |
    v
[file_manipulation.py] -- transcribe (Whisper) -> classify (GPT-4o) -> assign tier -> save Parquet
    |
    v
[data/labeled/*.parquet] -- structured complaint data
    ^
    |
[app.py] -- Streamlit chatbot queries all Parquet files via DuckDB
```

1. **watch.py**: Background watcher that detects new audio files and triggers the pipeline
2. **file_manipulation.py**: Core logic for transcription, classification, priority mapping, and storage
3. **app.py**: Streamlit chatbot with LangChain ReAct agent, optional voice input/output

---

## Prerequisites

- **Operating System**: Windows 10/11 with WSL2, or native Linux/macOS
- **Python**: 3.10 or 3.11 (3.11 recommended)
- **GPU (optional but recommended)**: NVIDIA GPU with CUDA support for fast Whisper transcription
- **CPU alternative**: Whisper runs on CPU with INT8 quantization (slower but fully functional)
- **Internet access**: Required for OpenAI API calls (GPT-4o, TTS)

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

If you have an NVIDIA GPU and want fast transcription, install the CUDA toolkit inside your conda environment:

```bash
conda install -c conda-forge cudatoolkit=12.1 -y
```

If you do NOT have an NVIDIA GPU, skip this step. See the [GPU vs CPU Configuration](#gpu-vs-cpu-configuration) section for how to configure Whisper for CPU.

---

## API Keys

This project requires API keys from two services. Both use the same OpenAI API key.

### OpenAI API Key

Used for: GPT-4o (chatbot agent + complaint classification) and TTS (voice responses)

1. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (it starts with `sk-`)

### Hugging Face Token

Used for: Downloading the Whisper model weights on first run

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

With your conda environment activated (`conda activate chatbot_env`), install all required packages:

### Core Dependencies

```bash
pip install --break-system-packages \
    streamlit \
    duckdb \
    pandas \
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

### Original Complaint Data

The project expects a labeled Parquet file in the `data/labeled/` directory. This file contains the original consumer complaint dataset with topic labels and priority scores.

```bash
mkdir -p data/labeled
```

Place your labeled Parquet file(s) in `data/labeled/`. The schema should include these columns:

```
Date received, Product, Sub-product, Issue, Sub-issue,
Consumer complaint narrative, Company, State, Tags,
Company response to consumer, Consumer disputed?, clean_narrative,
word_count, lda_topic, lda_topic_label, bert_topic, bert_topic_label,
category, year_month, year, month, date_dt, volume_score, growth_score,
is_unresolved, severity_score, recency_score, length_score, danger_boost,
priority_score, priority_tier
```

### Audio Recordings Directory

The file watcher monitors a directory for new audio recordings. By default this is set to:

```
/mnt/c/.../Documents/Sound Recordings
```

To change this, edit the `PATH` variable in `watch.py`:

```python
PATH = "/path/to/your/audio/recordings"
```

On WSL2, Windows paths are accessible under `/mnt/c/`. For example, `C:\Users\YourName\Documents` becomes `/mnt/c/Users/YourName/Documents`.

### Karen Avatar Image

Place an image file called `karen.png` in the project root directory. This is used as the chatbot avatar in the Streamlit UI.

---

## Running the Project

The project has two components that run simultaneously in separate terminals.

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

When a new recording appears, it will automatically transcribe, classify, and store it.

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
|-- file_manipulation.py      # Transcription, classification, storage logic
|-- karen.png                 # Chatbot avatar image
|-- .env                      # API keys (not committed to git)
|-- .gitignore
|
|-- data/
|   |-- labeled/
|   |   |-- complaints_labeled.parquet    # Original labeled dataset
|   |   |-- recording_20260423_143207.parquet  # Auto-generated from audio
|   |   |-- recording_20260423_151022.parquet  # Auto-generated from audio
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
| DuckDB | latest | SQL engine that queries Parquet files directly |
| Apache Parquet | -- | Columnar data storage format |
| Streamlit | latest | Web application framework for the chat UI |
| Plotly | latest | Interactive chart generation |
| Pandas | latest | Dataframe display formatting |
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

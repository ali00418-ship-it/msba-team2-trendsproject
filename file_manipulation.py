import pandas as pd
import os
from datetime import datetime, date
import duckdb
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import time

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("WHISP_KEY")


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
    Writes each transcription as its own small parquet file in parquet_dir.
    Files are named by timestamp so they never collide.
    The chatbot queries all of them at once via a glob: 'data/*.parquet'
    """
    os.makedirs(parquet_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(parquet_dir, f"recording_{timestamp}.parquet")

    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe = transcription.replace("'", "''")

    duckdb.sql(f"""
        COPY (
            SELECT
               TIMESTAMP'{today}' AS "Date received",
                '{safe}'       AS "{target_column}",
                'Unknown'      AS "Company",
                'MN'           AS "State"
        ) TO '{out_path}' (FORMAT PARQUET)
    """)

    print(f"Transcription saved to '{out_path}'.\n")


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


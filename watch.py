import file_manipulation as fm
import time

PARQUET_DIR = "data"
TARGET_COLUMN = "Consumer complaint narrative"

def main():
    PATH = "/mnt/c/Users/Raja/Documents/Sound Recordings"
    files = fm.get_existing_files(PATH)

    try:
        while True:
            result = fm.check_for_new_file(PATH, files)

            if result:
                print(f"New file {result[0]} found")
                transcription = fm.transcribe(audio_path=result[0])
                fm.append_transcription(transcription, PARQUET_DIR, TARGET_COLUMN)
                files = fm.get_existing_files(PATH)
            else:
                print("No new file found")

            time.sleep(20)

    except KeyboardInterrupt:
        print("\nShutting down watcher.")

main()

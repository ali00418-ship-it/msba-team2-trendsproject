import os
import time
import file_manipulation as fm

PARQUET_DIR = "data"
TARGET_COLUMN = "Consumer complaint narrative"

def is_file_ready(file_path, interval=2, retries=3):
    """
    Checks if a file is finished being written to by monitoring size stability.
    """
    last_size = -1
    stable_count = 0
    
    while stable_count < retries:
        try:
            current_size = os.path.getsize(file_path)
            # Windows placeholder files are often 0 bytes initially
            if current_size == last_size and current_size > 0:
                stable_count += 1
            else:
                stable_count = 0 # Reset if it's still growing
            
            last_size = current_size
        except OSError:
            # File might be locked or inaccessible momentarily
            return False
            
        time.sleep(interval)
    
    return True

def main():
    # Use the path you defined
    PATH = "/mnt/c/Users/Raja/Documents/Sound Recordings"
    files = fm.get_existing_files(PATH)

    try:
        while True:
            result = fm.check_for_new_file(PATH, files)

            if result:
                new_file_path = result[0]
                print(f"Detected potential new file: {new_file_path}. Waiting for recording to finish...")

                # --- THE FIX ---
                if is_file_ready(new_file_path):
                    print(f"File {new_file_path} is stable. Starting transcription.")
                    transcription = fm.transcribe(audio_path=new_file_path)
                    fm.append_transcription(transcription, PARQUET_DIR, TARGET_COLUMN)
                    # Update the baseline list only after successful processing
                    files = fm.get_existing_files(PATH)
                else:
                    print(f"File {new_file_path} is still active or empty. Skipping for now.")
                # ---------------
                
            else:
                print("No new file found")

            time.sleep(5)

    except KeyboardInterrupt:
        print("\nShutting down watcher.")

if __name__ == "__main__":
    main()
import pandas as pd
import os
from datetime import datetime

def clip(filename, n=1000):

    df = pd.read_csv(filename, n)

    df.to_csv('clipped.csv')


def get_existing_files(directory_path):
    """
    Scans a directory and returns a set containing all current filenames.
    """
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return set()

    # List comprehension to get files only (ignores subdirectories)
    return {f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))}


def check_for_new_file(directory_path, existing_files_set):
    """
    Compares the current directory contents against a known set of filenames.
    Returns: (full_file_path, date_added) if a new file is found, otherwise None.
    """
    if not os.path.exists(directory_path):
        return None

    # Get the current state of the directory
    current_files = {f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))}
    
    # Set difference: finds items in current_files that are NOT in existing_files_set
    new_files = current_files - existing_files_set
    
    if new_files:
        # Grab the first new file found
        new_file_name = new_files.pop()
        full_path = os.path.join(directory_path, new_file_name)
        
        # Get the timestamp. 'getmtime' (modification time) is generally the most 
        # reliable across different operating systems for detecting when a file finished copying.
        timestamp = os.path.getmtime(full_path)
        date_added = datetime.fromtimestamp(timestamp)
        
        return full_path, date_added

    # No new files were found
    return None
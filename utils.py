# utils.py

import os

def create_folder_if_not_exists(folder_name):
    """Create folder if it doesn't already exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

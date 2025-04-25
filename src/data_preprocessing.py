import os
import shutil

def organize_data(raw_path, organized_path):
    """Organize dataset into classes."""
    os.makedirs(organized_path, exist_ok=True)

    # Loop through original data and classify based on folder name
    for folder_name in os.listdir(raw_path):
        folder_path = os.path.join(raw_path, folder_name)

        if not os.path.isdir(folder_path):
            continue

        label = folder_name.capitalize()  # Make first letter uppercase (e.g., "Anthracite")

        # Create class folder if not exists
        class_dir = os.path.join(organized_path, label)
        os.makedirs(class_dir, exist_ok=True)

        # Move files into respective class folders
        for fname in os.listdir(folder_path):
            file_path = os.path.join(folder_path, fname)
            if os.path.isfile(file_path):
                shutil.copy(file_path, os.path.join(class_dir, fname))
    print("Data organized successfully.")

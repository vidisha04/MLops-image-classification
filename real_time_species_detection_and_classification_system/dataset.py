import io
import os
import shutil
from pathlib import Path
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm

# Path to your service account JSON file
SERVICE_ACCOUNT_FILE = "real_time_species_detection_and_classification_system/credentials.json"

# Scopes for Google Drive API
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def list_files_in_folder(service, folder_id):
    """
    List all files in a Google Drive folder.

    Args:
        service: Authenticated Google Drive service object.
        folder_id: The ID of the folder to list files from.

    Returns:
        A list of dictionaries containing file IDs, names, and mime types.
    """
    try:
        query = f"'{folder_id}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
        return results.get("files", [])
    except HttpError as error:
        print(f"An error occurred: {error}")
        return []


def download_file(service, file_id, file_name, output_dir):
    """
    Downloads a file from Google Drive using its file ID.

    Args:
        service: Authenticated Google Drive service object.
        file_id: The ID of the file to download.
        file_name: The name of the output file.
        output_dir: The directory to save the downloaded file.

    Returns:
        The path to the downloaded file.
    """
    try:
        output_path = Path(output_dir) / file_name
        os.makedirs(output_path.parent, exist_ok=True)

        request = service.files().get_media(fileId=file_id)
        with open(output_path, "wb") as file:
            downloader = MediaIoBaseDownload(file, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        return output_path

    except HttpError as error:
        print(f"An error occurred while downloading {file_name}: {error}")
        return None


def process_folder(service, folder_id, output_dir):
    """
    Recursively process and download all files in a Google Drive folder.

    Args:
        service: Authenticated Google Drive service object.
        folder_id: The ID of the folder to process.
        output_dir: The directory to save downloaded files.
    """
    files = list_files_in_folder(service, folder_id)
    total_files = len(files)
    if total_files == 0:
        print(f"No files found in folder: {output_dir}")
        return

    # Progress bar for the entire folder
    with tqdm(total=total_files, desc=f"Downloading folder: {Path(output_dir).name}", unit="file") as pbar:
        for file in files:
            file_name = file["name"]
            file_id = file["id"]
            mime_type = file["mimeType"]

            # Check if the item is a folder
            if mime_type == "application/vnd.google-apps.folder":
                new_output_dir = Path(output_dir) / file_name
                print(f"Entering folder: {file_name}")
                os.makedirs(new_output_dir, exist_ok=True)  # Create the folder structure
                process_folder(service, file_id, new_output_dir)
            else:
                # If the item is a file, download it
                download_file(service, file_id, file_name, output_dir)
            pbar.update(1)


def clean_existing_files(output_dir):
    """
    Delete existing files and folders in the specified output directory.

    Args:
        output_dir: The directory to clean.
    """
    paths_to_delete = ["test", "train", "sample_submission.csv"]
    for path in paths_to_delete:
        full_path = Path(output_dir) / path
        if full_path.is_dir():
            print(f"Deleting folder: {full_path}")
            shutil.rmtree(full_path)
        elif full_path.is_file():
            print(f"Deleting file: {full_path}")
            full_path.unlink()


if __name__ == "__main__":
    # Authenticate using the service account
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    drive_service = build("drive", "v3", credentials=creds)

    # Root folder ID from Google Drive
    folder_id = "1dwGzQwooU4PT642faBxBj_CS6OB_FiFE"  # Folder ID for data
    output_dir = "data/raw"

    # Clean up existing files and folders
    clean_existing_files(output_dir)

    print("Processing folder...")
    process_folder(drive_service, folder_id, output_dir)
    print("All files and folders downloaded successfully.")

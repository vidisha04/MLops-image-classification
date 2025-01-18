import io
import os
from pathlib import Path
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from google.cloud import storage
from tqdm import tqdm

# Path to your service account JSON file
SERVICE_ACCOUNT_FILE = "real_time_species_detection_and_classification_system/credentials.json"

# Scopes for Google Drive API
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Google Cloud Storage bucket name
BUCKET_NAME = "mlops_species_detection_data"

# Google Drive folder IDs
folder_ids = {
    'Amphibia': '1MpR8ZoI3QLnWbqisKovFbW_McE_o1--2',
    'Animalia': '1W59fH3UJbWLv3ANQmAf-Zch5sxSj6pTw',
    'Arachnida': '1OFrhaKiTjZ4MTMoTRW14VkmDYEVbEpAm',
    'Aves': '15el3Jpm_YncRul2Y1VLuMI6NQ9umME3B',
    'Fungi': '1JlVyX0d7VGuN4SRfOKda8OJg87m4I_iz',
    'Insecta': '1TCg21HAn88XqDZo7pFpyQ52phtYSvFa-',
    'Mammalia': '1DyEjx9cIOyeFVmn3of-KYWJRZWwYj04F',
    'Mollusca': '1cXIXC6GgwTJEnPWP1OZQXVJBWoWUp6un',
    'Plantae': '1XF5PtEEB7_m1ttxq_hk-9VwkEXl6oXpo',
    'Reptilia': '1FygC0DGD0caRCJOVhZPj_CvJvXLOCeWr'
}


def list_files_in_folder(service, folder_id):
    try:
        query = f"'{folder_id}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
        return results.get("files", [])
    except HttpError as error:
        print(f"An error occurred: {error}")
        return []

def upload_to_gcs(bucket_name, source_file, destination_blob):
    try:
        client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_FILE)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_filename(source_file)
        print(f"File {source_file} uploaded to {bucket_name}/{destination_blob}.")
    except Exception as error:
        print(f"Failed to upload {source_file} to GCS: {error}")

def download_file(service, file_id, file_name, bucket_name, destination_prefix):
    try:
        temp_file = Path("/tmp") / file_name
        os.makedirs(temp_file.parent, exist_ok=True)

        request = service.files().get_media(fileId=file_id)
        with open(temp_file, "wb") as file:
            downloader = MediaIoBaseDownload(file, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()

        # Upload to Google Cloud Storage
        gcs_path = f"{destination_prefix}/{file_name}" if destination_prefix else file_name
        upload_to_gcs(bucket_name, str(temp_file), gcs_path)
        temp_file.unlink()  # Remove the temporary file after upload
        return True

    except HttpError as error:
        print(f"An error occurred while downloading {file_name}: {error}")
        return False

def process_folder(service, folder_id, bucket_name, destination_prefix=""):
    files = list_files_in_folder(service, folder_id)
    total_files = len(files)
    if total_files == 0:
        print(f"No files found in folder: {destination_prefix}")
        return

    with tqdm(total=total_files, desc=f"Processing folder: {destination_prefix or 'root'}", unit="file") as pbar:
        for file in files:
            file_name = file["name"]
            file_id = file["id"]
            mime_type = file["mimeType"]

            if mime_type == "application/vnd.google-apps.folder":
                # Recursively process subfolders
                new_prefix = f"{destination_prefix}/{file_name}" if destination_prefix else file_name
                process_folder(service, file_id, bucket_name, new_prefix)
            else:
                # Process files
                download_file(service, file_id, file_name, bucket_name, destination_prefix)
            pbar.update(1)

if __name__ == "__main__":
    # Authenticate using the service account
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    drive_service = build("drive", "v3", credentials=creds)

    # Process each folder
    raw_dir = "raw"
    for folder_name, folder_id in folder_ids.items():
        print(f"Processing folder: {folder_name}")
        process_folder(drive_service, folder_id, BUCKET_NAME, destination_prefix=raw_dir)
        print(f"All files processed for {folder_name}\n")

    print("All files and folders processed successfully.")

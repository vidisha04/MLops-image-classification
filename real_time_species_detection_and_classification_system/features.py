from pathlib import Path
import os
from PIL import Image
import typer
from loguru import logger
from tqdm import tqdm
from torchvision import transforms
import torch
import pandas as pd
from google.cloud import storage
import tempfile

# Initialize Typer app
app = typer.Typer()

# Configure the logger
LOG_FILE = Path("logs/preprocessing.log")
os.makedirs(LOG_FILE.parent, exist_ok=True)
logger.add(LOG_FILE, rotation="500 MB", level="INFO", backtrace=True, diagnose=True)

# Label Mapping
LABEL_MAPPING = {
    "Amphibia": 0,
    "Animalia": 1,
    "Arachnida": 2,
    "Aves": 3,
    "Fungi": 4,
    "Insecta": 5,
    "Mammalia": 6,
    "Mollusca": 7,
    "Plantae": 8,
    "Reptilia": 9,
}

# Google Cloud Storage settings
GCS_BUCKET_NAME = "mlops_species_detection_data"
RAW_FOLDER = "raw"
PROCESSED_FOLDER = "processed"
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

# Preprocessing pipeline for ResNet-18
def get_resnet_preprocessing_pipeline():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def list_gcs_files(bucket, prefix):
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs if not blob.name.endswith("/")]

def process_and_upload(blob_name, pipeline, output_prefix, labels=None, is_test=False):
    try:
        # Download file to a temporary location
        blob = bucket.blob(blob_name)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            blob.download_to_filename(temp_file.name)
            temp_file_path = temp_file.name

        # Process the image
        with Image.open(temp_file_path).convert("RGB") as image:
            processed_image = pipeline(image)

        # Prepare the output blob name
        output_blob_name = blob_name.replace(RAW_FOLDER, PROCESSED_FOLDER).replace(".jpg", ".pt").replace(".jpeg", ".pt").replace(".png", ".pt")
        output_name = os.path.basename(output_blob_name)

        # Save processed image to a temporary file and upload to GCS
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            torch.save(processed_image, temp_file.name)
            bucket.blob(output_blob_name).upload_from_filename(temp_file.name)

        logger.info(f"Processed and uploaded: {output_blob_name}")

        # Add label if applicable
        if labels is not None:
            class_name = Path(blob_name).stem.split('_')[0]
            label = LABEL_MAPPING.get(class_name, None)
            if label is not None:
                labels.append({
                    "filename": output_name,
                    "label": label
                })
            else:
                logger.warning(f"Label not found for {blob_name}, skipping.")

    except Exception as e:
        logger.error(f"Failed to process {blob_name}: {e}")

def preprocess_gcs_folder(input_prefix, output_prefix, pipeline, labels_csv=None, is_test=False):
    labels = [] if not is_test else None

    files = list_gcs_files(bucket, input_prefix)
    if not files:
        logger.warning(f"No files found in GCS path: {input_prefix}")
        return

    for blob_name in tqdm(files, desc="Processing images", unit="file"):
        process_and_upload(blob_name, pipeline, output_prefix, labels, is_test)

    # Save labels CSV to GCS
    if labels_csv and labels:
        labels_df = pd.DataFrame(labels)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            labels_df.to_csv(temp_file.name, index=False)
            bucket.blob(labels_csv).upload_from_filename(temp_file.name)

        logger.info(f"Labels saved to: {labels_csv}")

@app.command()
def main(is_test:bool = False):
    """
    Preprocess images and upload them to GCS.
    
    Args:
        is_test (bool): If True, preprocess only test data. Default is False. (--is-test)
    """
    pipeline = get_resnet_preprocessing_pipeline()

    logger.info("Starting preprocessing pipeline...")

    try:
        if is_test:
            # Process testing data only
            preprocess_gcs_folder(
            f"test/{RAW_FOLDER}",
            f"test/{PROCESSED_FOLDER}",
            pipeline,
            is_test=True
            )
            logger.success("Test preprocessing complete. Processed data uploaded to GCS.")
        else: 
            # Process training data
            preprocess_gcs_folder(
                f"train/{RAW_FOLDER}",
                f"train/{PROCESSED_FOLDER}",
                pipeline,
                labels_csv="train/train_labels.csv",
                is_test=False
            )
            # Process testing data (to run the script in full mode)
            preprocess_gcs_folder(
                f"test/{RAW_FOLDER}",
                f"test/{PROCESSED_FOLDER}",
                pipeline,
                is_test=True
            )
            logger.success("Preprocessing complete. Processed data uploaded to GCS.")
    
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    app()

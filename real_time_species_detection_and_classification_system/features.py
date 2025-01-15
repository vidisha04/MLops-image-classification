from pathlib import Path
import os
from PIL import Image
import typer
from loguru import logger
from tqdm import tqdm
from torchvision import transforms
import torch
import pandas as pd
import sys

# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from real_time_species_detection_and_classification_system.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# Initialize Typer app
app = typer.Typer()

# Configure the logger
LOG_FILE = project_root / "logs" / "preprocessing.log"
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

def preprocess_images(input_dir: Path, output_dir: Path, pipeline, labels_csv=None, is_test=False):
    """
    Preprocess images from the input directory and save them to the output directory.

    Args:
        input_dir (Path): Path to the raw data directory (e.g., raw/train or raw/test).
        output_dir (Path): Path to the processed data directory (e.g., processed/train or processed/test).
        pipeline: Preprocessing pipeline to apply to each image.
        labels_csv: Path to save labels for the training dataset (only for train folder).
        is_test: Flag indicating whether the input directory is for test data.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting preprocessing for directory: {input_dir}")

    labels = []

    if is_test:
        # Test folder: Process files directly (no subdirectories)
        for file_name in tqdm(os.listdir(input_dir), desc="Processing test images", unit="file"):
            input_path = input_dir / file_name
            output_path = output_dir / file_name

            if input_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                try:
                    # Load and preprocess the image
                    image = Image.open(input_path).convert("RGB")
                    processed_image = pipeline(image)

                    # Save the preprocessed tensor
                    torch.save(processed_image, output_path.with_suffix(".pt"))
                    logger.info(f"Saved processed test file: {output_path.with_suffix('.pt')}")
                except Exception as e:
                    logger.error(f"Failed to process {input_path}: {e}")
            else:
                logger.warning(f"Skipped non-image file: {input_path}")
    else:
        # Train folder: Process subdirectories (classes)
        for class_name in sorted(os.listdir(input_dir)):
            class_dir = input_dir / class_name
            if not class_dir.is_dir():
                logger.warning(f"Skipping non-directory item: {class_dir}")
                continue

            logger.info(f"Processing class: {class_name}, Path: {class_dir}")
            label = LABEL_MAPPING.get(class_name, None)
            if label is None:
                logger.warning(f"Unknown class: {class_name}, skipping.")
                continue

            output_class_dir = output_dir / class_name
            os.makedirs(output_class_dir, exist_ok=True)
            logger.info(f"Output directory for class: {output_class_dir}")

            for file_name in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}", unit="file"):
                input_path = class_dir / file_name
                output_path = output_class_dir / file_name

                if input_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    try:
                        # Load and preprocess the image
                        image = Image.open(input_path).convert("RGB")
                        processed_image = pipeline(image)

                        # Save the preprocessed tensor
                        torch.save(processed_image, output_path.with_suffix(".pt"))
                        logger.info(f"Saved processed file: {output_path.with_suffix('.pt')}")

                        # Add folder name as prefix to filename
                        labels.append({
                            "filename": f"{class_name}/{file_name}".replace(Path(file_name).suffix, ".pt"),
                            "label": label
                        })

                    except Exception as e:
                        logger.error(f"Failed to process {input_path}: {e}")
                else:
                    logger.warning(f"Skipped non-image file: {input_path}")

    if labels_csv and not is_test:
        pd.DataFrame(labels).to_csv(labels_csv, index=False)
        logger.info(f"Labels saved to {labels_csv}")

    logger.info(f"Completed preprocessing for directory: {input_dir}")


@app.command()
def main(
    raw_train_path: Path = RAW_DATA_DIR / "train",
    raw_test_path: Path = RAW_DATA_DIR / "test",
    processed_train_path: Path = PROCESSED_DATA_DIR / "train",
    processed_test_path: Path = PROCESSED_DATA_DIR / "test",
    train_labels_csv: Path = PROCESSED_DATA_DIR / "train_labels.csv",
):
    logger.info("Starting the preprocessing pipeline.")

    pipeline = get_resnet_preprocessing_pipeline()

    try:
        # Process train dataset
        logger.info("Processing train dataset...")
        preprocess_images(
            raw_train_path,
            processed_train_path,
            pipeline,
            labels_csv=train_labels_csv,
            is_test=False
        )

        # Process test dataset
        logger.info("Processing test dataset...")
        preprocess_images(
            raw_test_path,
            processed_test_path,
            pipeline,
            is_test=True
        )

        logger.success("Preprocessing complete. Processed data saved in the processed directory.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    app()

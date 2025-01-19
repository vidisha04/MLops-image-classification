from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
from google.cloud import storage
import torch
from torchvision import models
import pandas as pd
import tempfile


# Typer app initialization
app = typer.Typer()

# GCS bucket details
GCS_BUCKET_NAME = "mlops_species_detection_data"
PROCESSED_TEST_PATH = "test/processed"
TRAINED_MODEL_PATH = "train/resnet18_model.pth"

# Initialize Google Cloud Storage client
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

def download_from_gcs(blob_name, local_path):
    """Download a file from GCS to a local path."""
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    logger.info(f"Downloaded {blob_name} to {local_path}")


def load_model(model_path):
    """
    Load the trained model from a state_dict.

    Args:
        model_path (str): Path to the model file.

    Returns:
        torch.nn.Module: The loaded model.
    """
    # Create the model architecture (e.g., ResNet-18)
    model = models.resnet18(pretrained=False, num_classes=10)  # Adjust num_classes as needed
    
    # Load the state_dict
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    # Set the model to evaluation mode
    model.eval()
    return model

@app.command()
def main(
    predictions_path: Path = Path("real_time_species_detection_and_classification_system/modeling/test_predictions.csv"),
):
    """
    Perform model inference and save predictions locally.
    
    Args:
        predictions_path (Path): Local path to save the predictions as a CSV file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Temporary paths for model and test data
        local_model_path = temp_dir_path / "resnet18_model.pth"
        local_test_folder = temp_dir_path / "test_processed"
        local_test_folder.mkdir(parents=True, exist_ok=True)

        try:
            # Download model
            logger.info("Downloading model from GCS...")
            download_from_gcs(TRAINED_MODEL_PATH, local_model_path)

            # Download processed test data
            logger.info("Downloading processed test data from GCS...")
            blobs = bucket.list_blobs(prefix=PROCESSED_TEST_PATH)
            test_files = [blob.name for blob in blobs if blob.name.endswith(".pt")]

            if not test_files:
                logger.error("No processed test data found in the GCS bucket!")
                return

            for file in tqdm(test_files, desc="Downloading test data"):
                local_file_path = local_test_folder / Path(file).name
                download_from_gcs(file, local_file_path)

            # Load the trained model
            logger.info("Loading the trained model...")
            model = load_model(local_model_path)

            # Perform inference on the test data
            logger.info("Performing inference on test data...")
            predictions = []
            for test_file in tqdm(local_test_folder.glob("*.pt"), desc="Making predictions"):
                # Load processed test image tensor
                input_tensor = torch.load(test_file)
                input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

                # Perform inference
                with torch.no_grad():
                    output = model(input_tensor)
                    predicted_class = torch.argmax(output, dim=1).item()

                # Collect prediction result
                predictions.append({
                    "filename": test_file.name,
                    "predicted_class": predicted_class
                })

            # Save predictions to CSV
            predictions_df = pd.DataFrame(predictions)
            predictions_df.to_csv(predictions_path, index=False)
            logger.success(f"Predictions saved to {predictions_path}")

        except Exception as e:
            logger.error(f"An error occurred during inference: {e}")
            raise

        # Temporary directory and its contents are automatically cleaned up here


if __name__ == "__main__":
    app()

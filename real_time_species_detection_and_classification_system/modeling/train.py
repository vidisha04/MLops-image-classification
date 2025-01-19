from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from loguru import logger
from tqdm import tqdm
import pandas as pd
import typer
from google.cloud import storage
import io

# Adjust sys.path to include the project root
import sys
project_root = Path(__file__).resolve().parents[2]  # Adjusting for 'modeling' subfolder
sys.path.append(str(project_root))

#from real_time_species_detection_and_classification_system.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

# Dataset Class
class GCSPreprocessedDataset(Dataset):
    def __init__(self, bucket_name: str, data_dir: str, labels_csv_blob: str):
        """
        Custom Dataset to load preprocessed tensors and labels from GCS.

        Args:
            bucket_name (str): Name of the GCS bucket.
            data_dir (str): Path in the bucket containing preprocessed .pt files.
            labels_csv_blob (str): Path in the bucket to the CSV file containing filenames and labels.
        """
        self.bucket_name = bucket_name
        self.data_dir = data_dir.rstrip("/")
        self.storage_client = storage.Client()

        # Download labels CSV
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(labels_csv_blob)
        labels_data = blob.download_as_bytes()
        self.labels_df = pd.read_csv(io.BytesIO(labels_data))
        self.file_names = self.labels_df["filename"].values
        self.labels = self.labels_df["label"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        label = self.labels[idx]

        # Load tensor from GCS
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(f"{self.data_dir}/{file_name}")
        tensor_data = blob.download_as_bytes()
        tensor = torch.load(io.BytesIO(tensor_data))
        return tensor, label


@app.command()
def main(
    gcs_bucket_name: str = "mlops_species_detection_data",
    train_data_dir: str = "train/processed",
    train_labels_csv: str = "train/train_labels.csv",
    model_output_path: str = "train/resnet18_model.pth",
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 0.001,
):
    """
    Train a ResNet-18 model using preprocessed tensors stored in GCS and save the trained model back to GCS.
    """
    logger.info("Starting model training...")

    # Load dataset and create DataLoader
    train_dataset = GCSPreprocessedDataset(gcs_bucket_name, train_data_dir, train_labels_csv)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load ResNet-18 model
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(set(train_dataset.labels)))  # Adjust output layer

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        logger.info(f"Epoch {epoch + 1}/{epochs}")
        for inputs, labels in tqdm(train_loader, desc="Training", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct / total
        logger.info(f"Epoch {epoch + 1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Save the trained model to GCS
    bucket = storage.Client().bucket(gcs_bucket_name)
    blob = bucket.blob(model_output_path)
    model_data = io.BytesIO()
    torch.save(model.state_dict(), model_data)
    model_data.seek(0)
    blob.upload_from_file(model_data, content_type="application/octet-stream")
    logger.success(f"Training complete. Model saved to {model_output_path} in GCS.")


if __name__ == "__main__":
    app()

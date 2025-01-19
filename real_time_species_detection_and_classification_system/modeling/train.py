import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from loguru import logger
from tqdm import tqdm
import pandas as pd
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import typer
from google.cloud import storage
import io


# Adjust sys.path to include the project root
project_root = Path(__file__).resolve().parents[2]  # Adjusting for 'modeling' subfolder
sys.path.append(str(project_root))

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


def train_model(cfg:DictConfig) -> None:
    """
    Train a ResNet-18 model using preprocessed tensors stored in GCS and save the trained model back to GCS.
    """
    logger.info("Starting model training...")

    # Load dataset and create DataLoader
    train_dataset = GCSPreprocessedDataset(cfg.gcs.bucket_name,
                                           cfg.gcs.train_data_dir,
                                           cfg.gcs.train_labels_csv)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)

    # Load ResNet-18 model
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(set(train_dataset.labels)))  # Adjust output layer

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prepare to log metrics
    metrics = {"epoch": [], "train_loss": [], "train_accuracy": []}

    # Training loop
    for epoch in range(cfg.training.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        logger.info(f"Epoch {epoch + 1}/{cfg.training.epochs}")
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

        # Save metrics
        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(epoch_loss)
        metrics["train_accuracy"].append(epoch_acc)

    # Save the trained model to GCS
    bucket = storage.Client().bucket(cfg.gcs.bucket_name)
    blob = bucket.blob(cfg.gcs.model_output_path)
    model_data = io.BytesIO()
    torch.save(model.state_dict(), model_data)
    model_data.seek(0)
    blob.upload_from_file(model_data, content_type="application/octet-stream")
    logger.success(f"Training complete. Model saved to {cfg.gcs.model_output_path} in GCS.")

    # Save metrics to interim directory
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(cfg.trainig.metrics_output_path, index=False)
    logger.info(f"Training metrics saved to {cfg.trainig.metrics_output_path}")


@hydra.main(version_base=None, config_path="conf", config_name="train_conf")
def train_model_with_hydra(cfg:DictConfig) -> None:
    """
    Train a Resnet-18 model with hydra configuration
    """
    train_model(cfg)

    
@app.command()
def train(batch_size: int = typer.Option(32, help="Batch size for training"),
          epochs: int = typer.Option(10, help="Number of epochs for training"),
          learning_rate: float = typer.Option(0.001, help="Learning rate for training"),
          metrics_output_path: str = typer.Option("data/interim/training_metric.csv", help="Metrics path")
          ) -> None:
    """Train a model on Species Images using Hydra configuration."""
    overrides = [
        f"train.batch_size={batch_size}",
        f"train.epochs={epochs}",
        f"train.learning_rate={learning_rate}",
        f"train.metrics_output_path={metrics_output_path}",
    ]
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path="conf")
    cfg = hydra.compose(config_name="train_conf", overrides=overrides)
    train_model(cfg)

if __name__ == "__main__":
    # If no typer subcommand is provided, fall back to running with Hydra
    if len(sys.argv) > 1 and sys.argv[1] in app.registered_commands: 
        app()
    else:  # Directly run Hydra-decorated function
        train_model_with_hydra()

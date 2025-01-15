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

# Adjust sys.path to include the project root
import sys
project_root = Path(__file__).resolve().parents[2]  # Adjusting for 'modeling' subfolder
sys.path.append(str(project_root))

from real_time_species_detection_and_classification_system.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

# Dataset Class
class PreprocessedDataset(Dataset):
    def __init__(self, data_dir: Path, labels_csv: Path):
        """
        Custom Dataset to load preprocessed tensors and labels.

        Args:
            data_dir (Path): Path to the directory containing preprocessed .pt files.
            labels_csv (Path): Path to the CSV file containing filenames and labels.
        """
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.file_names = self.labels_df["filename"].values
        self.labels = self.labels_df["label"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        label = self.labels[idx]
        tensor_path = self.data_dir / file_name
        tensor = torch.load(tensor_path)
        return tensor, label

@app.command()
def main(
    train_data_dir: Path = PROCESSED_DATA_DIR / "train",
    train_labels_csv: Path = PROCESSED_DATA_DIR / "train_labels.csv",
    model_output_path: Path = MODELS_DIR / "resnet18_model.pth",
    metrics_output_path: Path = INTERIM_DATA_DIR / "training_metrics.csv",
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 0.001,
):
    """
    Train a ResNet-18 model using preprocessed tensors and save the trained model and training metrics.
    """
    logger.info("Starting model training...")

    # Ensure output directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)

    # Load dataset and create DataLoader
    train_dataset = PreprocessedDataset(train_data_dir, train_labels_csv)
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

    # Prepare to log metrics
    metrics = {"epoch": [], "train_loss": [], "train_accuracy": []}

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

        # Save metrics
        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(epoch_loss)
        metrics["train_accuracy"].append(epoch_acc)

    # Save metrics to interim directory
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_output_path, index=False)
    logger.info(f"Training metrics saved to {metrics_output_path}")

    # Save the trained model
    torch.save(model.state_dict(), model_output_path)
    logger.success(f"Training complete. Model saved to {model_output_path}")


if __name__ == "__main__":
    app()

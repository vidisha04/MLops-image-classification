# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install necessary packages
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt requirements.txt
COPY data/interim data/interim
COPY pyproject.toml pyproject.toml
COPY real_time_species_detection_and_classification_system/ real_time_species_detection_and_classification_system/

# Copy the service account key file into the container
COPY real_time_species_detection_and_classification_system/credentials.json real_time_species_detection_and_classification_system/credentials.json

# Set the environment variable for Google Cloud authentication
ENV GOOGLE_APPLICATION_CREDENTIALS="real_time_species_detection_and_classification_system/credentials.json"

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# Install dependencies and package
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir


# Add entrypoint to run the training script
ENTRYPOINT ["python", "-u", "real_time_species_detection_and_classification_system/modeling/train.py"]

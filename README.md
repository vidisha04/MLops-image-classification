# Real-time Species Detection and Classification System

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Goal
The primary objective of this project is to develop a Real-Time Species Detection and Classification System for use in Scientific Research, with a focus on:
* Automating the classification of thousands of animal and plant images to aid ecological studies.
* Enabling real-time detection and monitoring of species distribution and migration patterns, facilitating data-driven insights into wildlife behavior and conservation efforts.

## Framework
The project will leverage PyTorch, which offers robust tools for building and training deep learning models. This framework will be integrated into the project as follows:

* Preprocessing and Augmentation: Using PyTorch's torchvision library to efficiently handle and preprocess large image datasets.
* Model Training: Building convolutional neural network (CNN) architectures using PyTorch for robust image classification.
* Real-Time Inference: Deploying trained models with PyTorch Mobile or ONNX for low-latency predictions in real-time scenarios.
* Command-Line Interface (CLI): Developing a user-friendly CLI to enable interaction with the application, including uploading images, triggering real-time inference, and receiving classification results. This will make the system accessible to researchers without requiring technical expertise.

## Dataset
The project will initially utilize the dataset described as follows:
* Training Data: 10,000 labeled images, subdivided into 10 folders, each representing a class.
* Classes include Amphibia, Animalia, Arachnida, Aves, Fungi, Insecta, Mammalia, Mollusca, Plantae, and Reptilia.

Test Data: 2,000 unlabeled images requiring classification.

Labels: Mapped as follows:
* Amphibia - 0
* Animalia - 1
* Arachnida - 2
* Aves - 3
* Fungi - 4
* Insecta - 5
* Mammalia - 6
* Mollusca - 7
* Plantae - 8
* Reptilia - 9

The dataset format includes a train folder for supervised learning and a test folder for evaluation and predictions.

*Source of the data: "[Kaggle, Deep Learning Practice - Image Classification](https://www.kaggle.com/competitions/deep-learning-practice-image-classification/data?select=train)"*

## Models
The project anticipates employing the following models:
1. Baseline Model: A simple CNN architecture to establish a baseline for classification accuracy.
2. Advanced Architectures:
* ResNet (Residual Networks): For improved feature extraction and handling complex patterns in the dataset.
* MobileNet: Lightweight architecture optimized for real-time inference.
3. Transfer Learning Models:Pre-trained models such as InceptionV3 or EfficientNet for leveraging existing knowledge and accelerating training.
4. Ensemble Methods: Combining predictions from multiple models to improve overall accuracy.

The selected models will be tuned and evaluated iteratively to achieve optimal performance for scientific research applications.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         real_time_species_detection_and_classification_system and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── real_time_species_detection_and_classification_system   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes real_time_species_detection_and_classification_system a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


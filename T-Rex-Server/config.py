import os

ROOT_DIR = r"C:\Users\dudyk\PycharmProjects\TrexTagging\T-Rex-Server"
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploads")
os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

DETECTION_DATA_PREFIX = "detection_data_"
CLASSIFICATION_DATASET_PREFIX = "classification_dataset_"
DETECTION_DATASET_PREFIX = "detection_dataset_"

WAITING_FOR_PROCESSING_PREFIX = "waiting_for_processing_"
PROCESSING_PREFIX = "processing_now_"
PROCESSED_PREFIX = "processed_"

CLASSIFICATION_MODELS_FOLDER_NAME = "classification_models"
DETECTION_MODELS_FOLDER_NAME = "detection_models"

REFERENCE_IMAGES_FOLDER_NAME = "reference_images"

WAITING_FOR_TRAINING_PREFIX = "waiting_for_training_"
TRAINING_PREFIX = "training_now_"
ERROR_WHILE_TRAINING_PREFIX = "error_while_training_"

TREX_API_TOKENS: list =  [
    "6e03a4ac2b0dd5b9173979236772beef",
    "6bb41d0e9b6e3eace1cff26d7fdeac84",
    "0a03c76b07ab92f13d0a44de1235284a",

    # "170160aec267001d7ff65491a21df3aa",
    # "5e612977c71ef886631498a0f2c03253",
    # "9a34dd503fad1d662c843d8f9f7baaf4",
    # "a401dbadc4cda03a7588d02cf7108f28",
]
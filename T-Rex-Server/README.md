
# FastAPI Server Project

## Overview
This project consists of a FastAPI server that handles various services related to data processing, model training, and dataset management. Below are brief descriptions of each service along with instructions on when to run them.

## Services

### Main Server (`main.py`)
The main server file that initializes the FastAPI application and handles API endpoints for various tasks such as file upload, dataset management, and model management.

**Run this service:**
- To start the FastAPI server and handle incoming API requests.
- To manage datasets, upload files, and handle model-related operations.

**Command to run:**
```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Detection Data Process Service (`detection_data_process_service.py`)
This service processes videos and images to extract detection data, which is later used for training detection models.

**Run this service:**
- When you have new video or image data that needs to be processed for detection tasks.
- To continuously monitor and process incoming detection data.

**Command to run:**
```sh
python detection_data_process_service.py
```

### Detection Training Service (`detection_training_service.py`)
This service handles the training of detection models using YOLO. It searches for new datasets and trains models based on the provided configurations.

**Run this service:**
- When you have new datasets that need to be used for training detection models.
- To continuously monitor and train new detection models as datasets become available.

**Command to run:**
```sh
python detection_training_service.py
```

### Classification Training Service (`classification_training_service.py`)
This service handles the training of classification models using ViT (Vision Transformer). It searches for new datasets and trains models based on the provided configurations.

**Run this service:**
- When you have new datasets that need to be used for training classification models.
- To continuously monitor and train new classification models as datasets become available.

**Command to run:**
```sh
python classification_training_service.py
```

## Usage
1. **Start the Main Server:** Run the command to start the FastAPI server. This will handle API requests related to file uploads, dataset management, and model management.
2. **Process Detection Data:** Run the detection data process service to process new video and image data for detection.
3. **Train Detection Models:** Run the detection training service to train new detection models as new datasets are available.
4. **Train Classification Models:** Run the classification training service to train new classification models as new datasets are available.

Ensure that each service is running in its own terminal or background process to enable seamless operation of the entire system.

## Requirements
- Python 3.7+
- FastAPI
- Uvicorn
- Torch
- YOLOv5 (Ultralytics)
- Vision Transformer (ViT)

## Installation
Install the required Python packages using pip:
```sh
pip install fastapi uvicorn torch ultralytics
```

## Configuration
Update the `config.py` file with the necessary configurations for paths, device settings, and other relevant parameters.

## Contributing
Feel free to submit issues or pull requests if you find any bugs or have suggestions for improvements.

## License
This project is licensed under the MIT License.

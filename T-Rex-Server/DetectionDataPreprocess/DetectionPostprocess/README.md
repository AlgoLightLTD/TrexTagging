
# Detection-Postprocess

This repository contains scripts for post-processing object detection results. The primary focus is on classifying objects within bounding boxes detected in images using a pre-trained PyTorch model.

## File Overview

### detection_postprocess.py

This script includes the following functions:

1. `classify_detection_bbx(image, bounding_boxes, model, idx_to_class, batch_size=1, resize_dim=(224, 224), device='cpu')`
    - Classifies objects within bounding boxes in an image using a given model.
    - **Parameters:**
      - `image`: Path to the image file, the image as a NumPy array, or a torch tensor.
      - `bounding_boxes`: List of bounding boxes (x1, y1, x2, y2).
      - `model`: PyTorch model for classification.
      - `idx_to_class`: Dictionary mapping class indices to class names.
      - `batch_size`: Number of images to process in a batch.
      - `resize_dim`: Dimensions to resize the bounding boxes.
      - `device`: Device to run the model on ('cpu' or 'cuda').
    - **Returns:**
      - `class_names`: List of class names for each bounding box.
      - `confidence_scores`: List of confidence scores for each bounding box.

2. `classify_detections_bbx(detection_results, classification_model, idx_to_class, classification_model_batch_size=10, device='cpu')`
    - Classifies objects within bounding boxes in detection results using a given classification model.
    - **Parameters:**
      - `detection_results`: List of detection results, each containing an image path and bounding boxes.
      - `classification_model`: PyTorch model for classification.
      - `idx_to_class`: Dictionary or list mapping class indices to class names.
      - `classification_model_batch_size`: Number of images to process in a batch.
      - `device`: Device to run the model on ('cpu' or 'cuda').
    - **Returns:**
      - The function modifies the input `detection_results` in place, adding class names and confidence scores.

## Setup

### Prerequisites

- Python 3.x
- PyTorch
- torchvision
- PIL
- numpy

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Detection-Postprocess.git
   cd Detection-Postprocess
   ```

2. Install the required packages:
   ```
   pip install torch torchvision pillow numpy
   ```

## Usage

### Classifying Bounding Boxes

1. Import the required functions:
   ```python
   from detection_postprocess import classify_detection_bbx, classify_detections_bbx
   ```

2. Prepare your model and data:
   ```python
   import torch
   from torchvision import models

   # Load a pre-trained model
   model = models.resnet50(pretrained=True)
   model.eval()

   # Define the mapping from class indices to class names
   idx_to_class = {0: 'class0', 1: 'class1', ...}

   # Define the image and bounding boxes
   image_path = 'path/to/your/image.jpg'
   bounding_boxes = [(50, 50, 150, 150), (200, 200, 300, 300)]
   ```

3. Classify objects within the bounding boxes:
   ```python
   class_names, confidence_scores = classify_detection_bbx(image_path, bounding_boxes, model, idx_to_class)
   print(class_names, confidence_scores)
   ```

### Classifying Detection Results

1. Define your detection results:
   ```python
   detection_results = [
       {
           "image_path": "path/to/image1.jpg",
           "boxes": [(50, 50, 150, 150), (200, 200, 300, 300)]
       },
       ...
   ]
   ```

2. Classify objects within the detection results:
   ```python
   classify_detections_bbx(detection_results, model, idx_to_class)
   print(detection_results)
   ```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

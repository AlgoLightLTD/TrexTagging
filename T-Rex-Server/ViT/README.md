
# Vision Transformer (ViT) Training Project

This repository contains the code to train a Vision Transformer (ViT) model on a custom dataset using PyTorch.

## Contents

- `ViT_training.py`: Main script for training the Vision Transformer model.
- `ViT_training_utils.py`: Utility functions for data loading, processing, and plotting metrics.
- `ViT_training_usage.py`: Example script demonstrating how to use the training script with augmentations.
- `vit_usage.py`: Example script demonstrating how to use a trained ViT model for inference.

## Setup

1. Clone the repository:

```sh
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment and activate it:

```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
```

3. Install the required packages:

```sh
pip install -r requirements.txt
```

## Training

To train the Vision Transformer model, you can use the provided `ViT_training_usage.py` script. Adjust the parameters as needed.

```sh
python ViT_training_usage.py
```

### Important Parameters

- `images_folder_paths`: List of paths to the folders containing the training images.
- `train_transform`: Transformations to apply to the training images.
- `TRAIN_BATCH_SIZE`: Batch size for training.
- `TEST_BATCH_SIZE`: Batch size for validation.
- `training_folder_path`: Path to save the training outputs (models, logs, etc.).
- `use_partial_dataset`: Boolean flag to use a partial dataset.
- `dataset_percentage`: Percentage of the dataset to use if `use_partial_dataset` is `True`.
- `freeze_backbone`: Boolean flag to freeze the backbone of the model during training.
- `weighted_class`: Boolean flag to use class weights.
- `balanced`: Boolean flag to balance the dataset.
- `CUDA_DEVICE_IDS`: List of CUDA device IDs to use for training.

## Inference

To perform inference with a trained ViT model, use the `vit_usage.py` script. Ensure the model path is correctly set.

```sh
python vit_usage.py
```

## Utility Functions

The `ViT_training_utils.py` file contains several utility functions for data processing and metric plotting:

- `ImageDataset`: Custom dataset class for loading and processing images.
- `set_memory_limits`: Set memory limits for specific GPU devices.
- `save_split_indices`: Save split indices to a JSON file.
- `load_split_indices`: Load split indices from a JSON file.
- `compute_class_weights`: Compute class weights for balancing the dataset.
- `plot_metrics`: Plot and save bar graphs of metrics.
- `plot_confusion_matrix`: Plot and save confusion matrices.
- `load_dataset`: Load dataset and create DataLoader objects for training and validation.

## Example Usage

### Training

The following example demonstrates how to use the `ViT_training_usage.py` script for training:

```python
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ViT_training import train_vit_model
from ViT_training_utils import set_memory_limits

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-30, 30), shear=(-15, 15), p=0.5),
    A.Equalize(p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.MotionBlur(blur_limit=7, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

images_folder_paths = ["./dataset/crops_by_classes"]
weighted_class = False
augmented = True
use_partial_dataset, dataset_percentage = True, 0.02
freeze_backbone = True
training_folder_path = "./ViT_training"
os.makedirs(training_folder_path, exist_ok=True)
CUDA_DEVICE_IDS = [11, 12]
MEMORY_FRACTIONS = [1.0, 1.0]
TRAIN_BATCH_SIZE = 5500
TEST_BATCH_SIZE = 5500
set_memory_limits(CUDA_DEVICE_IDS, MEMORY_FRACTIONS)

train_vit_model(images_folder_paths=images_folder_paths,
                train_transform=train_transform,
                TRAIN_BATCH_SIZE=TRAIN_BATCH_SIZE, TEST_BATCH_SIZE=TEST_BATCH_SIZE,
                training_folder_path=training_folder_path,
                use_partial_dataset=use_partial_dataset, dataset_percentage=dataset_percentage,
                freeze_backbone=freeze_backbone, 
                weighted_class=weighted_class,
                balanced=True,
                CUDA_DEVICE_IDS=CUDA_DEVICE_IDS
            )
```

### Inference

The following example demonstrates how to use the `vit_usage.py` script for inference:

```python
import torch

DEVICE = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('./models/ViT_augmented/vit_model.pt')

# Generate a random input tensor with the shape (N, 3, 224, 224)
# N is batch size
x = torch.rand(4, 3, 224, 224)
y = model(x)
print(y)
```

## Acknowledgements

This project uses the Vision Transformer (ViT) model from the `timm` library.

## License

This project is licensed under the MIT License.

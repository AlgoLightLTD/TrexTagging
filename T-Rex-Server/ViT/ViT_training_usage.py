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
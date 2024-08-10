import os
import torch

from detection_data_process_pipeline import detection_data_process_by_video_path
from detection_data_process_pipeline_utils import create_reference_image_from_crops, draw_detection_result


DEVICE = torch.device(f'cuda:11' if torch.cuda.is_available() else 'cpu')


video_path: str = "/raid/NehorayProjects/Ben/Vehicle Detection/bb_dataset/videos/2/videos/2022/מרץ/ch01_00000000000000000.mp4"
save_outputs_folder: str = "/raid/NehorayProjects/Ben/Vehicle Detection/trex_pipeline_test"
trex2_api_tokens: list =  ["6e03a4ac2b0dd5b9173979236772beef",]
yolo_model_path: str = "/raid/NehorayProjects/Ben/Vehicle Detection/weights/best_regev.pt"

classification_model_path: str = "/raid/NehorayProjects/Ben/Vehicle Detection/weights/vit_model.pt"
classification_idx_to_class_name = {
    0:'bicycle',
    1:'bus',
    2:'car',
    3:'jeep',
    4:'minibus',
    5:'minivan',
    6:'motorcycle',
    7:'taxi',
    8:'tender',
    9:'truck'
}

CLASSIFICATION_BATCH_SIZE: int = 50,

crops_source_folder: str = "/raid/NehorayProjects/Ben/Vehicle Classification/dataset/crops_by_classes"

TREX_EVERY_N_FRAMES: float = 24

YOLO_BATCH_SIZE: int = 400  # YOLO batch size for processing frames
YOLO_EVERY_N_FRAMES: float = 24  # Number of frames to skip before running YOLO again

N_CROPS_FROM_EACH_CLASS: int = 5  # Number of random images to select from each class

USE_BGS: bool = True
BG_N_SAMPLES: int = 240  # Number of images to keep in the queue
ADD_BG_EVERY: float = 2.0
UPDATE_BG_EVERY: float = 120  # Frequency to update background

verbose: bool = True

reference_image_path = os.path.join(save_outputs_folder, 'reference_image.png')  # Path to save reference image
reference_image, reference_bounding_boxes = create_reference_image_from_crops(crops_source_folder, N_CROPS_FROM_EACH_CLASS, reference_image_path)  # Create reference image from crops

non_overlapping_detections, overlapping_detections, detection_results = detection_data_process_by_video_path(
    video_path=video_path,
    save_outputs_folder=save_outputs_folder,
    trex2_api_tokens=trex2_api_tokens,
    yolo_model_path=yolo_model_path,
    device=DEVICE,
    classification_model_path=classification_model_path,
    classification_idx_to_class_name=classification_idx_to_class_name,
    CLASSIFICATION_BATCH_SIZE=CLASSIFICATION_BATCH_SIZE,

    reference_images_data=[{"image_path":reference_image_path, "boxes":reference_bounding_boxes}],

    TREX_EVERY_N_FRAMES=TREX_EVERY_N_FRAMES,
    YOLO_BATCH_SIZE=YOLO_BATCH_SIZE,
    YOLO_EVERY_N_FRAMES=YOLO_EVERY_N_FRAMES,
    USE_BGS=USE_BGS,
    BG_N_SAMPLES=BG_N_SAMPLES,
    ADD_BG_EVERY=ADD_BG_EVERY,
    UPDATE_BG_EVERY=UPDATE_BG_EVERY,
    verbose=verbose
)

draw_detection_result(detection_results[0])["image"].save("t.png")
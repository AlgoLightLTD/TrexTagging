import time
import glob
import os
import json
import torch
import uuid

from DetectionDataPreprocess.TRex.TRex import pickle_save

from config import *

# Assume DetectionDataProcessService functions are imported and available
from DetectionDataPreprocess.detection_data_process_pipeline import detection_data_process_by_video_path, detection_data_process_by_image_paths
from pipline_res_to_yolo import convert_trex_results_to_yolo

DEVICE = torch.device(f'cuda:11' if torch.cuda.is_available() else 'cpu')

def process_videos(videos: list, base_save_outputs_folder: str, config):
    for video_path in videos:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_outputs_folder = os.path.join(base_save_outputs_folder, video_name)
        os.makedirs(save_outputs_folder, exist_ok=True)
        _, _, detection_results = detection_data_process_by_video_path(
            video_path=video_path,
            save_outputs_folder=save_outputs_folder,
            trex2_api_tokens=TREX_API_TOKENS,
            yolo_model_path=config['yolo_model_path'],
            device=DEVICE,
            classification_model_path=config['classification_model_path'],
            classification_idx_to_class_name=config['classification_idx_to_class_name'],
            CLASSIFICATION_BATCH_SIZE=config.get('CLASSIFICATION_BATCH_SIZE', (50 if torch.cuda.is_available() else 1)),
            reference_images_data=config.get('reference_images_data', None),
            TREX_EVERY_N_FRAMES=config.get('TREX_EVERY_N_FRAMES', 24),
            YOLO_BATCH_SIZE=config.get('YOLO_BATCH_SIZE', (400 if torch.cuda.is_available() else 1)),
            YOLO_EVERY_N_FRAMES=config.get('YOLO_EVERY_N_FRAMES', 24),
            USE_BGS=config.get('USE_BGS', False),
            BG_N_SAMPLES=config.get('BG_N_SAMPLES', 240),
            ADD_BG_EVERY=config.get('ADD_BG_EVERY', 2.0),
            UPDATE_BG_EVERY=config.get('UPDATE_BG_EVERY', 120),
            verbose=config.get('verbose', True)
        )

        # save the detection results to pickle file
        pickle_save(detection_results, os.path.join(save_outputs_folder, 'detection_results.pkl'))


def process_images(images, save_outputs_folder, config):
    classification_idx_to_class_name=config.get('classification_idx_to_class_name')
    if type(classification_idx_to_class_name) == dict:
        classification_idx_to_class_name = list(classification_idx_to_class_name.values())

    _, _, detection_results = detection_data_process_by_image_paths(
        images_paths=images,
        save_outputs_folder=save_outputs_folder,
        trex2_api_tokens=TREX_API_TOKENS,
        yolo_model_path=config['yolo_model_path'],
        device=DEVICE,
        classification_model_path=config.get('classification_model_path'),
        classification_idx_to_class_name=classification_idx_to_class_name,
        CLASSIFICATION_BATCH_SIZE=config.get('CLASSIFICATION_BATCH_SIZE', (50 if torch.cuda.is_available() else 1)),
        reference_images_data=config.get('reference_images_data'),
        TREX_EVERY_N_FRAMES=config.get('TREX_EVERY_N_FRAMES', 1),
        YOLO_BATCH_SIZE=config.get('YOLO_BATCH_SIZE', (400 if torch.cuda.is_available() else 1)),
        YOLO_EVERY_N_FRAMES=config.get('YOLO_EVERY_N_FRAMES', 1),
        USE_BGS=config.get('USE_BGS', False),
        BG_N_SAMPLES=config.get('BG_N_SAMPLES', 240),
        ADD_BG_EVERY=config.get('ADD_BG_EVERY', 2.0),
        UPDATE_BG_EVERY=config.get('UPDATE_BG_EVERY', 120),
        verbose=config.get('verbose', True)
    )

    # save the detection results to pickle file
    pickle_save(detection_results, os.path.join(save_outputs_folder, 'detection_results.pkl'))

def search_folders():
    """
    Search for folders matching the specified pattern and process them.
    """
    waiting_for_processing_data_paths_pattern = os.path.join(UPLOAD_DIR, "**", WAITING_FOR_PROCESSING_PREFIX+"*")
    processing_now_data_paths_pattern = os.path.join(UPLOAD_DIR, "**", PROCESSING_PREFIX+"*")
    need_pipeline_data_paths = glob.glob(waiting_for_processing_data_paths_pattern, recursive=True) + glob.glob(processing_now_data_paths_pattern, recursive=True)
    
    for need_pipeline_data_path in need_pipeline_data_paths:
        info_json_path = os.path.join(need_pipeline_data_path, 'process_pipeline_data.json')
        
        if os.path.exists(info_json_path):
            with open(info_json_path, 'r') as f:
                config = json.load(f)

            if WAITING_FOR_PROCESSING_PREFIX in need_pipeline_data_path:
                new_need_pipeline_data_path = need_pipeline_data_path.replace(WAITING_FOR_PROCESSING_PREFIX, PROCESSING_PREFIX)
                os.rename(need_pipeline_data_path, new_need_pipeline_data_path)
                need_pipeline_data_path = new_need_pipeline_data_path

            # Assuming 'video_paths' and 'image_paths' are specified in the config
            video_paths = glob.glob(os.path.join(need_pipeline_data_path, '**', '*.mp4'), recursive=True)
            image_paths = glob.glob(os.path.join(need_pipeline_data_path, '**', '*.jpg'), recursive=True) + \
                          glob.glob(os.path.join(need_pipeline_data_path, '**', '*.png'), recursive=True)

            if video_paths:
                process_videos(video_paths, os.path.join(need_pipeline_data_path, "videos"), config)

            if image_paths:
                process_images(image_paths, os.path.join(need_pipeline_data_path, "all_images"), config)
            

            #### Convert T-Rex results to YOLO format ####
            classification_idx_to_class_name = config['classification_idx_to_class_name']
            if type(classification_idx_to_class_name) == list:
                trex_output_classes_to_idx = {name:i for i, name in enumerate(classification_idx_to_class_name)}
            elif type(classification_idx_to_class_name) == dict:
                trex_output_classes_to_idx = {name:i for i, name in classification_idx_to_class_name.items()}
            
            detection_dataset_save_path = need_pipeline_data_path.replace(PROCESSING_PREFIX+DETECTION_DATA_PREFIX, DETECTION_DATASET_PREFIX) # Creating new folder for detection dataset
            convert_trex_results_to_yolo(need_pipeline_data_path, detection_dataset_save_path, trex_output_classes_to_idx, verbose=True) # Convert T-Rex results to YOLO format

            #### get some final detetion dataset analysis ####

            #### Save the dataset info ####
            dataset_json_file_path = os.path.join(detection_dataset_save_path, "info.json")
            with open(dataset_json_file_path, "w") as info_file:
                json.dump({"title": config.get('title', None), "description": config.get('description', None), "class_names": [class_name for _, class_name in trex_output_classes_to_idx.items()]}, info_file, indent=4)

            #### Move the processed folder to processed folder ####
            processed_folder_path = need_pipeline_data_path.replace(PROCESSING_PREFIX, PROCESSED_PREFIX)
            os.rename(need_pipeline_data_path, processed_folder_path)

def run_service():
    """
    Run the service indefinitely.
    """
    while True:
        search_folders()
        time.sleep(10)  # Wait for 10 seconds before searching again

if __name__ == "__main__":
    run_service()
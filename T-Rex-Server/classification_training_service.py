
import shutil
import time
import glob
import os
import json

import torch
from ViT.ViT_training import train_vit_model
from ViT.ViT_training_utils import set_memory_limits
from ViT.ViT_training_utils import train_transform

from config import CLASSIFICATION_DATASET_PREFIX, CLASSIFICATION_MODELS_FOLDER_NAME, TRAINING_PREFIX, UPLOAD_DIR, WAITING_FOR_TRAINING_PREFIX, ERROR_WHILE_TRAINING_PREFIX

CUDA_DEVICE_IDS = [0,]
MEMORY_FRACTIONS = [1.0,]
TRAIN_BATCH_SIZE = 5500
TEST_BATCH_SIZE = 5500

def search_folders():
    """
    Search for folders matching the specified pattern and process them.
    """
    waiting_training_models_pattern = os.path.join(UPLOAD_DIR, "**", CLASSIFICATION_MODELS_FOLDER_NAME, WAITING_FOR_TRAINING_PREFIX+"*.json")
    need_training_models = glob.glob(waiting_training_models_pattern, recursive=True)
    for need_training_model in need_training_models:
        #### Get model id from json file ####
        model_id = os.path.basename(need_training_model).replace(WAITING_FOR_TRAINING_PREFIX, '').replace('.json', '')

        #### Load the info.json file ####
        with open(need_training_model, 'r') as f:
            info = json.load(f)
        # info should contain the following keys: dataset_id
        dataset_id = info['dataset_id']
        user_id = info['user_id']

        #### get dataset path ####
        dataset_path_pattern = os.path.join(UPLOAD_DIR, user_id, CLASSIFICATION_DATASET_PREFIX+dataset_id)
        dataset_paths = glob.glob(dataset_path_pattern, recursive=True)
        # check if exists
        if not dataset_paths or not os.path.exists(dataset_paths[0]):
            # rename to error while training
            error_while_training_model = need_training_model.replace(WAITING_FOR_TRAINING_PREFIX, ERROR_WHILE_TRAINING_PREFIX)
            os.rename(need_training_model, error_while_training_model)

            # write that dataset was not found to the json file in a new key names message
            info['message'] = 'Dataset not found'
            with open(error_while_training_model, 'w') as f:
                json.dump(info, f)
            continue
    
        dataset_path = os.path.join(dataset_paths[0], "dataset")

        #### rename to training and update status ####
        training_model = need_training_model.replace(WAITING_FOR_TRAINING_PREFIX, TRAINING_PREFIX)
        os.rename(need_training_model, training_model)
        info['status'] = 'training'
        with open(training_model, 'w') as f:
            json.dump(info, f)
        
        #### Train the model ####
        images_folder_paths = [dataset_path]

        training_folder_path = os.path.join(UPLOAD_DIR, user_id, CLASSIFICATION_MODELS_FOLDER_NAME, model_id)
        os.makedirs(training_folder_path, exist_ok=True)

        if torch.cuda.is_available():
            set_memory_limits(CUDA_DEVICE_IDS, MEMORY_FRACTIONS)

        best_model_path = train_vit_model(images_folder_paths=images_folder_paths,
                        train_transform=train_transform,
                        TRAIN_BATCH_SIZE=TRAIN_BATCH_SIZE, TEST_BATCH_SIZE=TEST_BATCH_SIZE,
                        training_folder_path=training_folder_path,
                        use_partial_dataset=False, dataset_percentage=1.0,
                        freeze_backbone=False, 
                        weighted_class=False,
                        balanced=True,
                        CUDA_DEVICE_IDS=CUDA_DEVICE_IDS
                    )
        
        #### save best model at final location ####
        final_model_path = os.path.join(UPLOAD_DIR, user_id, CLASSIFICATION_MODELS_FOLDER_NAME, model_id+".pt")
        shutil.copy(best_model_path, final_model_path) # copy model

        #### update status to trained and remove prefix ####
        info['status'] = ''
        with open(training_model, 'w') as f:
            json.dump(info, f)
        final_model = training_model.replace(TRAINING_PREFIX, '')
        os.rename(training_model, final_model)

def run_service():
    """
    Run the service indefinitely.
    """
    while True:
        search_folders()
        time.sleep(10)  # Wait for 10 seconds before searching again

if __name__ == "__main__":
    run_service()
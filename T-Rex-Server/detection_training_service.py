import shutil
import time
import glob
import os
import json

import torch
from ultralytics import YOLO

from config import DETECTION_DATASET_PREFIX, DETECTION_MODELS_FOLDER_NAME, TRAINING_PREFIX, UPLOAD_DIR, WAITING_FOR_TRAINING_PREFIX, ERROR_WHILE_TRAINING_PREFIX

CUDA_DEVICES = [0,]
BATCH_SIZE = 50
EPOCHS = 1

def search_folders():
    """
    Search for folders matching the specified pattern and process them.
    """
    waiting_training_models_pattern = os.path.join(UPLOAD_DIR, "**", DETECTION_MODELS_FOLDER_NAME, WAITING_FOR_TRAINING_PREFIX+"*.json")
    need_training_models = glob.glob(waiting_training_models_pattern, recursive=True)
    for need_training_model in need_training_models:
        #### Get model id from json file ####
        model_id = os.path.basename(need_training_model).replace(WAITING_FOR_TRAINING_PREFIX, '').replace('.json', '')
        model_training_dir = os.path.join(os.path.dirname(need_training_model), model_id)
        os.makedirs(model_training_dir, exist_ok=True)  

        #### Load the info.json file of model ####
        with open(need_training_model, 'r') as f:
            info = json.load(f)
        # info should contain the following keys: dataset_id
        dataset_id = info['dataset_id']
        user_id = info['user_id']
        yolo_base_model = info['yolo_base_model']

        #### get dataset path ####
        dataset_path_pattern = os.path.join(UPLOAD_DIR, user_id, DETECTION_DATASET_PREFIX+dataset_id)
        dataset_paths = glob.glob(dataset_path_pattern, recursive=True)
        # check if exists
        if not dataset_paths or not os.path.exists(dataset_paths[0]):
            # rename to error while training
            error_while_training_model = need_training_model.replace(WAITING_FOR_TRAINING_PREFIX, ERROR_WHILE_TRAINING_PREFIX)
            os.rename(need_training_model, error_while_training_model)

            # write that dataset was not found to the json file in a new key named message
            info['message'] = 'Dataset not found'
            with open(error_while_training_model, 'w') as f:
                json.dump(info, f)
            continue
    
        dataset_path = dataset_paths[0]

        #### load dataset info.json file ####
        with open(os.path.join(dataset_path, 'info.json'), 'r') as f:
            dataset_info = json.load(f)
        class_names = dataset_info['class_names']

        #### rename to training and update status ####
        training_model = need_training_model.replace(WAITING_FOR_TRAINING_PREFIX, TRAINING_PREFIX)
        os.rename(need_training_model, training_model)
        info['status'] = 'training'
        with open(training_model, 'w') as f:
            json.dump(info, f)
        
        #### Train the YOLO model using ultralytics ####
        detection_model_path = os.path.join(UPLOAD_DIR, user_id, DETECTION_MODELS_FOLDER_NAME, model_id+'.pt')
        base_dataset_path = dataset_path
        num_classes = len(class_names)

        #### create the yaml file with relevant data ####
        dataset_yaml_content = f"""path: {base_dataset_path}

# Relative paths to images and labels
train: images/train  # Relative path to training images
val: images/val  # Relative path to validation images

# Class names
names:\n"""
        for i, class_name in enumerate(class_names):
            dataset_yaml_content += f"  {i}: {class_name}\n"

        dataset_yaml_content += f"\n# Number of classes\nnc: {num_classes}\n"

        dataset_yaml_path = os.path.join(base_dataset_path, 'dataset.yaml')
        with open(dataset_yaml_path, 'w') as f:
            f.write(dataset_yaml_content)

        # Create a YOLO model
        model = YOLO(yolo_base_model.lower())  # Load a pre-trained model

        # Train the model with explicit paths and parameters
        model.train(
            imgsz=640,
            epochs=EPOCHS,
            data=dataset_yaml_path,  # Path to the YAML file
            device=','.join(map(str, CUDA_DEVICES)) if torch.cuda.is_available() else 'cpu',  # Use CUDA if available
            batch=BATCH_SIZE,
            project=model_training_dir
        )

        # Save the trained model
        shutil.copy(os.path.join(model_training_dir, 'train', 'weights', 'best.pt'), detection_model_path)

        #### rename info to processed ####
        processed_model = training_model.replace(TRAINING_PREFIX, '')
        os.rename(training_model, processed_model)

        #### update status to trained and remove prefix ####
        info['status'] = ''
        with open(processed_model, 'w') as f:
            json.dump(info, f)
        
def run_service():
    """
    Run the service indefinitely.
    """
    while True:
        search_folders()
        time.sleep(10)  # Wait for 10 seconds before searching again

if __name__ == "__main__":
    run_service()

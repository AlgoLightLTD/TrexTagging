from glob import glob  
import os  
import shutil  
import pickle  
import uuid  
from PIL import Image  
from tqdm import tqdm  

#### Function to convert detection results to YOLO format ####
def convert_to_yolo_format(detection_results, images_output_path, labels_output_path, trex_output_classes_to_idx):
    """
    Convert detection results to YOLO format and save images and labels.

    Parameters:
        detection_results (list): List of detection results.
        images_output_path (str): Path to save images.
        labels_output_path (str): Path to save labels.
        trex_output_classes_to_idx (dict): Mapping from class names to indices.

    Returns:
        None
    """
    for result in detection_results:  # Iterate over each detection result
        image_path = result['image_path']  # Get image path
        boxes = result['boxes']  # Get bounding boxes
        classes = result['classes']  # Get classes

        if not classes:  # If no classes, skip
            continue

        # Generate new image filename with UUID
        original_filename = os.path.basename(image_path)  # Get original filename
        unique_id = uuid.uuid4()  # Generate unique ID
        new_image_filename = f"{os.path.splitext(original_filename)[0]}_{unique_id}{os.path.splitext(original_filename)[1]}"  # Create new filename
        new_image_path = os.path.join(images_output_path, new_image_filename)  # Create new image path

        # Copy image to the YOLO dataset folder with new name
        image_path = image_path.replace(".mp4", "")  # Remove ".mp4" extension from image path
        shutil.copy(image_path, new_image_path)  # Copy image to new path

        # Get the image size
        with Image.open(new_image_path) as img:  # Open the image
            img_width, img_height = img.size  # Get image dimensions

        # Create corresponding label file
        label_filename = os.path.splitext(new_image_filename)[0] + '.txt'  # Create label filename
        label_filepath = os.path.join(labels_output_path, label_filename)  # Create label file path

        with open(label_filepath, 'w') as f:  # Open label file for writing
            for box, cls in zip(boxes, classes):  # Iterate over boxes and classes
                if cls is None:  # If class is None, skip
                    continue

                # Convert box coordinates to YOLO format
                x_center = (box[0] + box[2]) / 2.0  # Calculate x center
                y_center = (box[1] + box[3]) / 2.0  # Calculate y center
                width = box[2] - box[0]  # Calculate width
                height = box[3] - box[1]  # Calculate height

                # Normalize coordinates
                x_center /= img_width  # Normalize x center
                y_center /= img_height  # Normalize y center
                width /= img_width  # Normalize width
                height /= img_height  # Normalize height

                # Write to label file
                f.write(f"{trex_output_classes_to_idx[cls]} {x_center} {y_center} {width} {height}\n")  # Write label data

#### Function to convert T-Rex results to YOLO format ####
def convert_trex_results_to_yolo(base_folder_path, output_base_path, trex_output_classes_to_idx, train_ratio=0.8, verbose=False):
    """
    Convert T-Rex pipeline results to YOLO format for training and validation.

    Parameters:
        base_folder_path (str): Path to the base folder of results.
        output_base_path (str): Base path for YOLO dataset output.
        trex_output_classes_to_idx (dict): Mapping from class names to indices.
        train_ratio (float): Ratio of data to use for training. Default is 0.8.

    Returns:
        None
    """
    train_images_path = os.path.join(output_base_path, 'images', 'train')  # Path for training images
    val_images_path = os.path.join(output_base_path, 'images', 'val')  # Path for validation images
    train_labels_path = os.path.join(output_base_path, 'labels', 'train')  # Path for training labels
    val_labels_path = os.path.join(output_base_path, 'labels', 'val')  # Path for validation labels

    # Create directories if they don't exist
    os.makedirs(train_images_path, exist_ok=True)  # Create training images directory
    os.makedirs(val_images_path, exist_ok=True)  # Create validation images directory
    os.makedirs(train_labels_path, exist_ok=True)  # Create training labels directory
    os.makedirs(val_labels_path, exist_ok=True)  # Create validation labels directory


    for detection_results_path in tqdm(glob(os.path.join(base_folder_path, "**", 'detection_results.pkl'), recursive=True), disable=not verbose):  # Find all detection result files
        with open(detection_results_path, 'rb') as f:  # Open detection result file
            detection_results = pickle.load(f)  # Load detection results

        # Split results into training and validation sets
        split_idx = int(len(detection_results) * train_ratio)  # Calculate split index
        train_results = detection_results[:split_idx]  # Training results
        val_results = detection_results[split_idx:]  # Validation results

        # Convert to YOLO format
        convert_to_yolo_format(train_results, train_images_path, train_labels_path, trex_output_classes_to_idx)  # Convert training results
        convert_to_yolo_format(val_results, val_images_path, val_labels_path, trex_output_classes_to_idx)  # Convert validation results

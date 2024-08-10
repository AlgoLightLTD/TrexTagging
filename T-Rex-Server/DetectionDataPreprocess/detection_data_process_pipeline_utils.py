from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import io
import pickle
import cv2
from ultralytics import YOLO
from tqdm import tqdm

import torch
start_cuda = torch.cuda.Event(enable_timing=True, blocking=True)
finish_cuda = torch.cuda.Event(enable_timing=True, blocking=True)

def load_images_yolo(image_paths):
    """
    Load images from given paths and convert them to RGB format.

    Parameters:
        image_paths (list): List of paths to the images to be loaded.

    Returns:
        list: List of images in RGB format as numpy arrays.
    """
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")  # Open and convert image to RGB
        images.append(np.array(image))  # Append image as numpy array to the list
    return images

def process_yolo_bbx(yolo_result, detection_conf_threshold=0.2):
    """
    Process YOLO results to extract bounding boxes with a confidence score above a threshold.

    Parameters:
        yolo_result (object): YOLO result object containing detection data.
        detection_conf_threshold (float): Confidence threshold for filtering detections. Default is 0.2.

    Returns:
        list: List of bounding boxes (x1, y1, x2, y2) and class idx that meet the confidence threshold.
    """
    to_ret = []
    for bbox, cls, score in zip(yolo_result.boxes.xyxy.cpu().numpy(), yolo_result.boxes.cls.cpu().numpy(), yolo_result.boxes.conf.cpu().numpy()):
        if score >= detection_conf_threshold:  # Check if detection confidence is above threshold
            x1, y1, x2, y2 = map(int, bbox)  # Convert bounding box coordinates to integers
            to_ret.append(((x1, y1, x2, y2), cls))  # Append bounding box to the result list
    return to_ret

def detect_objects_batch_yolo(image_paths, model, device, imgsz=640):
    """
    Detect objects in a batch of images using a YOLO model.

    Parameters:
        image_paths (list): List of paths to the images to be processed.
        model (object): YOLO model used for object detection.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
        imgsz (int): Size of the input images for the model. Default is 640.

    Returns:
        list: List of dictionaries containing image paths and their corresponding bounding boxes with class name for each bbx.
    """
    images = load_images_yolo(image_paths)  # Load images from paths
    names = model.names
    results = model(images, imgsz=imgsz, device=device)  # Perform object detection using YOLO model

    to_ret = []
    for image_path, result in zip(image_paths, results):
        bbxs = process_yolo_bbx(result)  # Process YOLO result to extract bounding boxes
        to_ret.append({
            "image_path": image_path,  # Original image path
            "bounding_boxes": [bbx[0] for bbx in bbxs],  # Detected bounding boxes
            "classes": [names[int(bbx[1])] for bbx in bbxs]
        })
    return to_ret

def run_yolo_on_image_paths(yolo_model_path: str, run_yolo_frames_paths: list, yolo_results_folder_path: str = None, YOLO_BATCH_SIZE: int = 400, device: str = 'cpu', verbose: bool = False):
    """
    Run YOLO model on a list of image paths, saving and loading results as needed.

    Parameters:
        yolo_model_path (str): Path to the YOLO model.
        run_yolo_frames_paths (list): List of image paths to run YOLO on.
        yolo_results_folder_path (str): Directory to save YOLO results. If None, results are not saved.
        YOLO_BATCH_SIZE (int): Batch size for processing images with YOLO. Default is 400.
        device (str): Device to run the YOLO model on. Default is 'cpu'.
        verbose (bool): If True, display a progress bar. Default is False.

    Returns:
        list: List of YOLO results.
    """
    os.makedirs(yolo_results_folder_path, exist_ok=True)  # Create directory if it doesn't exist

    yolo_model = YOLO(yolo_model_path)  # Initialize YOLO model
    yolo_results = []  # Initialize YOLO results list
    batch_paths = []  # Initialize batch paths list

    n_images = len(run_yolo_frames_paths)  # Total number of images

    for idx, image_path in tqdm(enumerate(run_yolo_frames_paths), total=n_images, disable= not verbose):
        # Determine result file path if saving results
        result_file_path = os.path.join(yolo_results_folder_path, os.path.basename(image_path).rsplit('.')[0] + '.pkl') if yolo_results_folder_path else None

        if result_file_path and os.path.exists(result_file_path):  # Check if result already exists
            yolo_results.append(pickle_load(result_file_path))  # Load existing result
            continue  # Skip to the next image
        else:
            batch_paths.append(image_path)  # Add to batch if result does not exist

        # Process batch if it reaches the batch size limit or it's the last batch
        if len(batch_paths) >= YOLO_BATCH_SIZE or (idx == n_images - 1 and len(batch_paths) > 0):
            batch_results = detect_objects_batch_yolo(batch_paths, yolo_model, device)  # Get YOLO results
            yolo_results.extend(batch_results)  # Extend results list with batch results

            # Save batch results individually if a save path is provided
            if yolo_results_folder_path:
                for result in batch_results:
                    result_file_path = os.path.join(yolo_results_folder_path, os.path.basename(result["image_path"]).rsplit('.')[0] + '.pkl')
                    pickle_save(result, result_file_path)  # Save result

            batch_paths = []  # Reset batch paths list

    return yolo_results  # Return all YOLO results

def create_reference_image(objects_paths):
    """
    Create a reference image by arranging a list of object images vertically.

    Parameters:
        objects_paths (List[str]): List of file paths to the object images.

    Returns:
        Tuple[Image.Image, List[List[int]]]: Reference image and positions of each object in the reference image.
    """
    # Load and convert all images to RGB
    crop_images = [Image.open(path).convert("RGB") for path in objects_paths]  # Open each image and convert to RGB
    
    # Sort images by height in descending order (tallest first)
    crop_images.sort(key=lambda im: im.height, reverse=True)  # Sort images by their height

    max_width = max(im.width for im in crop_images)  # Find the maximum width among all images
    total_height = sum(im.height for im in crop_images)  # Calculate the total height of all images

    # Initialize the reference image with a conservative estimate
    reference_image = Image.new("RGB", (max_width, total_height))  # Create a new blank image with the max width and total height

    # Initialize positions list
    positions = []  # List to store positions of each image in the reference image

    y_offset = 0  # Vertical offset to place the next image
    x_offset = 0  # Horizontal offset to place the next image
    row_height = 0  # Height of the current row

    # Loop through each cropped image and place it in the reference image
    for im in crop_images:
        if x_offset + im.width > max_width:
            # Move to the next row if the current image exceeds the max width
            y_offset += row_height  # Increment the vertical offset by the row height
            x_offset = 0  # Reset horizontal offset
            row_height = 0  # Reset row height

        reference_image.paste(im, (x_offset, y_offset))  # Paste the image at the current offset
        positions.append([x_offset, y_offset, x_offset + im.width, y_offset + im.height])  # Store the position of the image
        x_offset += im.width  # Increment horizontal offset by the image width
        row_height = max(row_height, im.height)  # Update the row height

    # Crop the reference image to remove unused space
    reference_image = reference_image.crop((0, 0, max_width, y_offset + row_height))  # Crop the image to the used area

    return reference_image, positions  # Return the reference image and positions

def select_random_images(folder_path, n):
    """
    Selects n random images from the specified folder.

    Args:
        folder_path (str): Path to the folder containing images.
        n (int): Number of random images to select.

    Returns:
        list: List of file paths for the selected images.
    """
    all_files = os.listdir(folder_path)  # List all files in the folder
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]  # Filter image files
    if len(image_files)==0:
        return []
    n_images = n if n<len(image_files) else len(image_files)
    selected_images = random.sample(image_files, n_images)  # Randomly select n images
    selected_image_paths = [os.path.join(folder_path, image) for image in selected_images]  # Create full paths for the selected images
    return selected_image_paths  # Return the list of selected image paths

def create_reference_image_from_crops(crops_folder_path: str, n: int = 5, reference_image_path: str = "reference_image.png"):
    ### Collect paths of random images from each class subfolder: ###
    objects_paths = []  # Initialize objects paths list
    for class_folder in os.listdir(crops_folder_path):  # Loop through class folders
        class_folder_path = os.path.join(crops_folder_path, class_folder)  # Get class folder path
        if os.path.isdir(class_folder_path):  # Check if path is directory
            objects_paths.extend(select_random_images(class_folder_path, n))  # Add random images to list

    ### Create the reference image: ###
    if len(objects_paths)>0:
        reference_image, bounding_boxes = create_reference_image(objects_paths)  # Create reference image
        reference_image.save(reference_image_path)  # Save reference image
        return reference_image, bounding_boxes
    return None, None

def draw_bounding_boxes(image_input, bounding_boxes, bounding_boxes_confidence, class_list: list = None, class_confidence: list = None, yolo_classes: list = None, split_idx: int = None):
    """
    Draw bounding boxes with confidence scores on the image and return it.

    Parameters:
        image_input (str or np.ndarray): Path to the image file or the image as a NumPy array.
        bounding_boxes (List[Tuple[int, int, int, int]]): List of bounding boxes with coordinates (x1, y1, x2, y2).
        bounding_boxes_confidence (List[float]): List of confidence scores for each bounding box.
        class_list (List[str], optional): List of class names for each bounding box.
        class_confidence (List[float], optional): List of confidence scores for each class.

    Returns:
        PIL.Image.Image: Image with drawn bounding boxes and confidence scores.
    
    Raises:
        AssertionError: If the length of bounding_boxes and bounding_boxes_confidence is not the same.
                        If class_list and class_confidence are provided, their length must match the length of bounding_boxes.
        TypeError: If image_input is not a file path or a NumPy array.
    """
    # Validate the lengths of input lists
    assert len(bounding_boxes) == len(bounding_boxes_confidence), "bounding_boxes and bounding_boxes_confidence must have the same length"  # Check if bounding_boxes and bounding_boxes_confidence have the same length
    assert (class_list is None and class_confidence is None) or (class_list and class_confidence), "Both class_list and class_confidence must be provided together"  # Ensure class_list and class_confidence are both provided or both None
    if class_list and class_confidence:
        assert len(class_list) == len(class_confidence) == len(bounding_boxes), "class_list, class_confidence, and bounding_boxes must have the same length"  # Check if class_list, class_confidence, and bounding_boxes have the same length when provided

    # Load the image
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")  # Open the image file and convert it to RGB
    elif isinstance(image_input, np.ndarray):
        if image_input.shape[-1] == 1:
            image_input = image_input.squeeze(-1)  # Remove the last dimension if it is 1 (grayscale image)
        image = Image.fromarray(image_input.astype('uint8')).convert("RGB")  # Convert the NumPy array to an RGB image
    else:
        raise TypeError("image_input should be either a file path or a NumPy array")  # Raise an error if image_input is not a valid type
    
    draw = ImageDraw.Draw(image)  # Create a drawing object to draw on the image
    font = ImageFont.load_default()  # Load the default font for drawing text
    
    # Draw bounding boxes and confidence scores on the image
    outline_color = "red"
    for idx in range(len(bounding_boxes)):
        bbox, confidence = bounding_boxes[idx], bounding_boxes_confidence[idx]  # Get the current bounding box and confidence score
        x1, y1, x2, y2 = bbox  # Unpack the bounding box coordinates

        #### set bbx color ####
        if split_idx and idx>=split_idx:
            outline_color = "green"
        
        #### Draw bbx and conf ####
        draw.rectangle([(x1, y1), (x2, y2)], outline=outline_color, width=2)  # Draw the bounding box with a red outline
        draw.text((x1, y1 - 10), f"{confidence:.2f}", fill="red", font=font)  # Draw the confidence score above the bounding box
        
        #### write classes from classifier and YOLO ####
        if class_list is not None and class_confidence is not None:
            class_name, class_conf = class_list[idx], class_confidence[idx]  # Get the current class name and class confidence score
            draw.text((x1, max(y1 - 20,0)), f"{class_name}, {class_conf:.2f}", fill="red", font=font)  # Draw the class name and confidence score above the bounding box
        if yolo_classes:
            yolo_class = yolo_classes[idx]
            draw.text((x1, max(y1 - 30,0)), f"YOLO: {yolo_class}", fill="red", font=font)
    
    return image  # Return the image with drawn bounding boxes and confidence scores

def draw_detection_result(detection_result: dict):
    """
    Show the results with bounding boxes and confidence scores.

    Parameters:
        detection_result (List[dict]): Dictionry containing detection results with the following keys:
            - 'image' or 'image_path' (str or np.ndarray): Path to the image file or the image as a NumPy array.
            - 'boxes' (List[Tuple[int, int, int, int]]): List of bounding boxes.
            - 'scores' (List[float]): List of confidence scores for each bounding box.
            - 'classes' (List[str], optional): List of class names for each bounding box.
            - 'classes_confidence' (List[float], optional): List of confidence scores for each class.
            - 'yolo_classes' (List[str], optional): List of classes detected by YOLO
    
    Returns:
        dict: A dictionry with 'image_path' and 'image' (image with drawn bounding boxes and confidence scores).
    """
    
    # Check that detection_result contains either 'image' or 'image_path' and required keys
    required_keys = {'boxes', 'scores'}  # Define the required keys
    has_image_key = 'image' in detection_result or 'image_path' in detection_result  # Check for 'image' or 'image_path' key
    has_required_keys = required_keys.issubset(detection_result.keys())  # Check if all required keys are present
    
    assert has_image_key, "detection_result must contain 'image' or 'image_path'"  # Assert presence of 'image' or 'image_path'
    assert has_required_keys, "detection_result must contain 'boxes' and 'scores'"  # Assert presence of required keys
    
    # Validate the optional keys if they are present
    if 'classes' in detection_result:  
        assert 'classes_confidence' in detection_result, "'classes_confidence' must be present if 'classes' is provided"  # Check for 'classes_confidence' if 'classes' is provided
    if 'classes_confidence' in detection_result:
        assert 'classes' in detection_result, "'classes' must be present if 'classes_confidence' is provided"  # Check for 'classes' if 'classes_confidence' is provided
    
    # Draw bounding boxes on the image
    if 'image_path' in detection_result:  
        image_input = detection_result['image_path']  # Use 'image_path' if available
    else:
        image_input = detection_result['image']  # Otherwise, use 'image'
        
    bounding_boxes = detection_result['boxes']  # Extract bounding boxes
    confidence_scores = detection_result['scores']  # Extract confidence scores
    class_list = detection_result.get('classes', None)  # Extract class list if available
    class_confidence = detection_result.get('classes_confidence', None)  # Extract class confidence if available
    yolo_classes = detection_result.get('yolo_classes', None)
    split_idx = detection_result.get('split_idx', None)  # Extract class confidence if available
    
    image_with_boxes = draw_bounding_boxes(image_input, bounding_boxes, confidence_scores, class_list, class_confidence, yolo_classes, split_idx=split_idx)  # Draw bounding boxes on the image
    
    return {"image_path": detection_result.get('image_path', None), "image": image_with_boxes}  # Return list of results with bounding boxes

def draw_detection_results(detection_results: list):
    """
    Draw detection results for a list of detections.

    Parameters:
        detection_results (list): List of detection results. Each detection result is a dictionary containing
                                  information such as image paths, bounding boxes, scores, etc.

    Returns:
        list: List of images with drawn detection results.
    """
    to_ret = []  # Initialize a list to store the resulting images with drawn detections

    # Loop through each detection result in the list
    for detection_result in detection_results:
        # Draw detection results on the image and append the resulting image to the list
        to_ret.append(draw_detection_result(detection_result))

    return to_ret  # Return the list of images with drawn detection results

def create_image_grid(image_data, axis: str='off'):
    """
    Organize images in a grid with titles and return the grid image.

    Parameters:
        image_data (List[List[Tuple[Union[np.ndarray, str, torch.Tensor, Image.Image], str]]]): 
            A list of lists of tuples, each containing an image and a title.
    
    Returns:
        PIL.Image.Image: Image containing the grid of input images with titles.
    """
    # Calculate the number of rows and columns in the grid
    num_rows = len(image_data)  # Number of rows
    num_cols = max(len(row) for row in image_data)  # Maximum number of columns
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))  # Create subplots for the grid
    axes = np.array(axes)  # Ensure axes is an array for consistency
    
    # Loop through the image data and plot each image in the grid
    for row_idx, row in enumerate(image_data):
        for col_idx, (image, title) in enumerate(row):
            ax = axes[row_idx, col_idx] if num_rows > 1 else axes[col_idx]  # Get the current axis
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")  # Load image from path
            elif isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy() if image.ndim == 3 else image.numpy()  # Convert torch tensor to numpy array
            elif isinstance(image, np.ndarray):
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB
                elif image.shape[-1] == 1:
                    image = np.squeeze(image, axis=-1)  # Squeeze single channel if necessary
            
            ax.imshow(image)  # Display image
            ax.set_title(title)  # Set image title
            ax.axis(axis)  # Hide axis
    
    # Remove empty subplots
    for ax in axes.flatten():
        if not ax.has_data():
            ax.axis("off")  # Hide empty axis
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    
    # Save the figure to a BytesIO buffer
    buf = io.BytesIO()  # Create an in-memory buffer
    plt.savefig(buf, format='png')  # Save the figure to the buffer
    buf.seek(0)  # Move the cursor to the beginning of the buffer
    grid_image = Image.open(buf)  # Open the image from the buffer
    plt.close(fig)  # Close the figure to free memory
    
    return grid_image  # Return the grid image

def pickle_save(obj, file_path):
    """
    Save an object to a file using pickle.
    
    Parameters:
    obj (any): The object to be pickled.
    file_path (str): The file path where the object will be saved.
    """
    with open(file_path, 'wb') as file:  # Open the file in write-binary mode
        pickle.dump(obj, file)  # Serialize and save the object

def pickle_load(file_path):
    """
    Load an object from a pickle file.
    
    Parameters:
    file_path (str): The file path from where the object will be loaded.
    
    Returns:
    any: The object that was loaded from the file.
    """
    with open(file_path, 'rb') as file:  # Open the file in read-binary mode
        obj = pickle.load(file)  # Deserialize and load the object
    return obj

def path_make_path_if_none_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def gtic():
    torch.cuda.synchronize()
    start_cuda.record()

def gtoc(pre_string='', verbose=True):
    ### Print: ###
    finish_cuda.record()
    torch.cuda.synchronize()
    total_time = start_cuda.elapsed_time(finish_cuda)
    if verbose:
        if pre_string != '':
            print(pre_string + ": %f msec." % total_time)
        else:
            print("Elapsed time: %f msec." % total_time)
    return total_time

def save_video_frames_to_disk(video_path: str, video_frames_folder: str, save_to_disk_every_n_frames: int = 1, verbose: bool = True):
    """
    Extract frames from a video and save each frame to disk.

    Parameters:
        video_path (str): Path to the video file.
        video_frames_folder (str): Folder to save the extracted frames.
        save_to_disk_every_n_frames (int): save every n frames.
        verbose (bool): Flag to indicate whether to print progress to the screen. Default is True.

    Returns:
        List[str]: List of paths to the saved frames.
    """
    cap = cv2.VideoCapture(video_path)  # Open the video file
    frame_count = 0  # Initialize frame counter
    frame_paths = []  # Initialize list to store frame paths

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames in the video

    if verbose:
        progress_bar = tqdm(total=total_frames)  # Initialize progress bar

    while frame_count<total_frames:
        if frame_count % save_to_disk_every_n_frames == 0:
            frame_path = os.path.join(video_frames_folder, f"frame_{frame_count}.jpg")  # Define frame path
            if not os.path.exists(frame_path):  # Check if frame path exists
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)  # Set the capture to the current frame position
                success, frame = cap.read()  # Read the first frame
                if success:
                    cv2.imwrite(frame_path, frame)  # Save frame to disk
                    frame_paths.append(frame_path)  # Add frame path to list
                elif verbose:
                    print(f"Error writing frame {frame_count} from {video_path}")
            else:
                frame_paths.append(frame_path)  # Add frame path to list
        else:
            frame_paths.append(None)

        frame_count += 1  # Increment frame counter

        if verbose:
            progress_bar.update(1)  # Update progress bar

    cap.release()  # Release video capture

    if verbose:
        progress_bar.close()  # Close progress bar

    return frame_paths  # Return list of frame paths

def process_image_paths_for_detection(image_paths: list, TREX_EVERY_N_FRAMES: int = 10, YOLO_EVERY_N_FRAMES: int = 30, verbose: bool = False):
    """
    Process a list of image paths to T-Rex and YOLO usage.

    Parameters:
        image_paths (List[str]): List of paths to the images.
        TREX_EVERY_N_FRAMES (int): Interval for processing frames with T-Rex. Default is 10.
        YOLO_EVERY_N_FRAMES (int): Interval for processing frames with YOLO. Default is 30.
        verbose (bool): Flag indicating whether to print progress and timing information. Default is False.

    Returns:
        List[Dict[str, Any]]: List of T-Rex image prompts.
        List[str]: list of image paths to process using YOLO
    """
    run_yolo_frames_paths = []  # Initialize list for YOLO frame paths
    images_prompts = []  # Initialize list for T-Rex image prompts

    for idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths), disable=not verbose):  # Loop through each image path
        # saving frame as a frame to use YOLO on
        if idx % YOLO_EVERY_N_FRAMES == 0:  # Check if frame should be processed by YOLO
            run_yolo_frames_paths.append(image_path)  # Add frame path to YOLO list

        # adding frame to t-rex processing list if needed
        if idx % TREX_EVERY_N_FRAMES == 0:
            images_prompts.append({  # Add T-Rex image prompt
                "image_path": image_path,
                "prompts": [{"prompt_image": run_yolo_frames_paths[-1]}],
                "idx": idx
            })

    return images_prompts, run_yolo_frames_paths  # Return list of T-Rex image prompts and background tensors

def combine_detections(detections_set_1: list, detections_set_2: list):
    """
    Combine two sets of detections, supporting different detection sets and maintaining current functionality.

    Parameters:
        detections_set_1 (list): First set of detection results.
        detections_set_2 (list): Second set of detection results.

    Returns:
        list: Combined list of detection results with split index.
    """
    # Sort the detection sets by index
    detections_set_1 = sorted(detections_set_1, key=lambda detection_result: detection_result["idx"])  # Sort set 1 by index
    detections_set_2 = sorted(detections_set_2, key=lambda detection_result: detection_result["idx"])  # Sort set 2 by index

    detection_results = []  # Initialize list to hold combined detection results
    idx_1, idx_2 = 0, 0  # Initialize indices for both detection sets

    while idx_1 < len(detections_set_1) and idx_2 < len(detections_set_2):
        detection_dict = {}  # Initialize dictionary to hold combined detection

        detection_set_1 = detections_set_1[idx_1]  # Get current detection from set 1
        detection_set_2 = detections_set_2[idx_2]  # Get current detection from set 2

        # Combine detections with the same index
        if detection_set_1["idx"] == detection_set_2["idx"]:
            for key in detection_set_1.keys():
                if isinstance(detection_set_1[key], np.ndarray):  # Convert arrays to lists
                    detection_set_1[key] = detection_set_1[key].tolist()
                    detection_set_2[key] = detection_set_2[key].tolist()
                
                if isinstance(detection_set_2[key], list):
                    detection_dict[key] = detection_set_1[key]  # Copy values from set 1
                    detection_dict["split_idx"] = len(detection_set_1[key])  # Set split index
                    detection_dict[key].extend(detection_set_2[key])  # Extend with values from set 2
                
                else:
                    detection_dict[key] = detection_set_2[key]  # Copy value from set 2 which should be the same as set 1

            detection_results.append(detection_dict)  # Add combined detection to results
            idx_1 += 1
            idx_2 += 1
        elif detection_set_1["idx"] < detection_set_2["idx"]:
            # Add detection from set 1 if it has a smaller index
            detection_results.append(detection_set_1)
            idx_1 += 1
        else:
            # Add detection from set 2 if it has a smaller index
            detection_results.append(detection_set_2)
            idx_2 += 1

    # Append any remaining detections from set 1
    while idx_1 < len(detections_set_1):
        detection_results.append(detections_set_1[idx_1])
        idx_1 += 1

    # Append any remaining detections from set 2
    while idx_2 < len(detections_set_2):
        detection_results.append(detections_set_2[idx_2])
        idx_2 += 1

    return detection_results  # Return combined detections
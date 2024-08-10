import math
import torch
import os

from DetectionDataPreprocess.TRex.TRex import trex_infer
from DetectionDataPreprocess.detection_data_process_pipeline_utils import save_video_frames_to_disk
from DetectionDataPreprocess.detection_data_process_pipeline_utils import *
from DetectionDataPreprocess.DetectionCleaning.detection_cleaning import clean_bounding_boxes_BGS, process_image_paths_to_bg_dict
from DetectionDataPreprocess.DetectionPostprocess.detection_postprocess import classify_detections_bbx
from DetectionDataPreprocess.DetectionAnalysis.detection_analysis import calculate_iou, split_detections_bbx_by_overlap

DEVICE = torch.device(f'cuda:11' if torch.cuda.is_available() else 'cpu')  # Set device to GPU if available, else CPU

def detection_data_process_by_image_paths(
        images_paths, save_outputs_folder: str, 
        trex2_api_tokens: list,
        yolo_model_path: str,
        device: str,

        classification_model_path: str = None, 
        classification_idx_to_class_name = None,

        reference_images_data: list = None,

        TREX_EVERY_N_FRAMES: int = 24,

        YOLO_BATCH_SIZE: int = 400,  # YOLO batch size for processing frames
        YOLO_EVERY_N_FRAMES: int = 24,  # Number of frames to skip before running YOLO again

        CLASSIFICATION_BATCH_SIZE = 50,
        
        USE_BGS: bool = True,
        BG_N_SAMPLES: int = 480,  # Number of images to keep in the queue
        ADD_BG_EVERY: int = 10,
        UPDATE_BG_EVERY: int = 120,  # Frequency to update background

        verbose: bool = False
    ):
    """
    Process a list of image paths using the T-Rex pipeline.

    Parameters:
        images_paths (list): List of image paths to process.
        save_outputs_folder (str): Directory to save the output results.
        trex2_api_tokens (list): List of API tokens for T-Rex.
        yolo_model_path (str): Path to the YOLO model.
        device (str): Device to run the models on ('cpu' or 'cuda').
        classification_model_path (str): Path to the classification model.
        classification_idx_to_class_name (dict): Dictionary mapping class indices to class names.
        reference_images_data (list): list which contains data about each reference image (dict with the keys "image_path" and "boxes" which represents the bounding boxes of the images).
        TREX_EVERY_N_FRAMES (int): Number of frames to process using T-Rex.
        YOLO_BATCH_SIZE (int): Batch size for processing frames using YOLO.
        YOLO_EVERY_N_FRAMES (int): Number of frames to skip before running YOLO again.
        CLASSIFICATION_BATCH_SIZE (int): Batch size for classification.
        USE_BGS (bool): Whether to use background subtraction.
        BG_N_SAMPLES (int): Number of images to keep in the background queue.
        ADD_BG_EVERY (int): Frequency to add background samples.
        UPDATE_BG_EVERY (int): Frequency to update the background.
        verbose (bool): Whether to print verbose output.

    Returns:
        tuple: Non-overlapping detections, overlapping detections, and all detection results (each detection's bbx are ordered with "split_idx" which indicates the transition from none overlapping to overlapping).
    """
    assert isinstance(images_paths, list), "images_paths must be a list of strings"  # Ensure images_paths is a list of strings
    
    if USE_BGS:
        if verbose: print(f"Building BG tensors")
        bg_tensor_folder = os.path.join(save_outputs_folder, 'bg_tensors')  # Path to save background tensors
        os.makedirs(bg_tensor_folder, exist_ok=True)  # Create directory if it doesn't exist
        bg_tensors = process_image_paths_to_bg_dict(images_paths, bg_tensor_folder, ADD_BG_EVERY, UPDATE_BG_EVERY, BG_N_SAMPLES, verbose=verbose)  # Process image paths to background tensor dictionary

    trex_images_prompts, run_yolo_frames_paths = process_image_paths_for_detection(images_paths, TREX_EVERY_N_FRAMES, YOLO_EVERY_N_FRAMES, verbose=verbose)  # Process image paths for detection and get T-Rex prompts

    ### Get YOLO results: ###
    if verbose: print(f"Running YOLO on {len(run_yolo_frames_paths)} images")
    yolo_results_folder_path = os.path.join(save_outputs_folder, 'yolo_results')  # Path to save YOLO results
    yolo_results = run_yolo_on_image_paths(yolo_model_path, run_yolo_frames_paths, yolo_results_folder_path, YOLO_BATCH_SIZE, device, verbose)  # Run YOLO on image paths and get results

    ### Get YOLO result for each desired image (build prompts for T-Rex) ###
    if verbose: print(f"Setting T-Rex prompts")
    for trex_image_prompts in trex_images_prompts:  # Loop through T-Rex image prompts
        valid_prompts = []  # Initialize valid prompts list
        for trex_image_prompt in trex_image_prompts['prompts']:  # Loop through T-Rex prompts
            for yolo_result in yolo_results:  # Loop through YOLO results
                if yolo_result["image_path"] == trex_image_prompt["prompt_image"] and len(yolo_result["bounding_boxes"]) > 0:  # Check if YOLO result matches prompt
                    trex_image_prompt["rects"] = yolo_result["bounding_boxes"]  # Add bounding boxes to prompt
                    valid_prompts.append(trex_image_prompt)  # Add prompt to valid prompts
                    break
        trex_image_prompts['prompts'] = valid_prompts  # Update prompts list

        if reference_images_data is not None:  # Check if reference image is available
            for reference_image_data in reference_images_data:
                trex_image_prompts['prompts'].append({  # Add reference image prompt
                    "prompt_image": reference_image_data["image_path"],
                    "rects": reference_image_data["boxes"]
                })
    
        if verbose and len(trex_image_prompts['prompts']) == 0:  # If verbose and no valid prompts
            print(f"No prompts for {trex_image_prompts['image_path']}")  # Print message
    
    trex_images_prompts = [trex_image_prompts for trex_image_prompts in trex_images_prompts if len(trex_image_prompts['prompts']) > 0]  # Filter out prompts with no valid prompts

    ### Use T-Rex: ###
    if verbose: print(f"Running T-Rex on {len(trex_images_prompts)} images")
    trex_save_path = os.path.join(save_outputs_folder, "trex_results")  # Path to save T-Rex results
    os.makedirs(trex_save_path, exist_ok=True)  # Create directory if it doesn't exist
    trex_results = trex_infer(trex_images_prompts, trex2_api_tokens, trex_save_path=trex_save_path, verbose=verbose)  # Get T-Rex results

    #### Cleaning T-Rex results which faild for some reason ####
    cleaned_trex_results = []
    for trex_result in trex_results:
        if len(trex_result["message"])>0 or "idx" not in trex_result.keys():
            if verbose:
                print(f'T-Rex encoutered error with {trex_result["image_path"]}: {trex_result["message"]}')
            continue
        else:
            cleaned_trex_results.append(trex_result)
    trex_results = cleaned_trex_results

    ### Use BGS ###
    if verbose: print(f"Using BGS to clean results")
    if USE_BGS:
        ### T-Rex result to BGS result ###
        for trex_result in trex_results:
            idx = trex_result["idx"]  # Get index from T-Rex result
            closest_key = min(bg_tensors.keys(), key=lambda k: abs(k - idx))  # Find closest key in background tensors
            trex_result["bg_tensor_key"] = closest_key  # Add closest key to T-Rex result

        ### Clean the bounding boxes using the BGS results: ###
        detection_results = clean_bounding_boxes_BGS(trex_results, bg_tensors)  # Clean bounding boxes
    else:
        detection_results = trex_results
    
    #### Assign YOLO classification results to T-Rex results ####
    if verbose: print(f"Combining YOLO class to T-Rex resutls")
    for detection_result in detection_results:  # Iterate through each T-Rex detection result
        detection_result["yolo_classes"] = []  # Initialize a list to store YOLO classes for each bounding box
        for yolo_result in yolo_results:  # Iterate through each YOLO detection result
            if yolo_result["image_path"] == detection_result["image_path"]:  # Check if the image paths match
                for bbox in detection_result["boxes"]:  # Iterate through each bounding box in the T-Rex result

                    max_iou = 0  # Initialize max_iou to track the highest IOU found
                    assigned_class = None  # Initialize assigned_class to store the class with the highest IOU

                    for idx, yolo_bbox in enumerate(yolo_result["bounding_boxes"]):  # Iterate through each YOLO bounding box
                        iou = calculate_iou(bbox, yolo_bbox)  # Calculate the IOU between T-Rex and YOLO bounding boxes
                        if iou > max_iou:  # Check if the current IOU is greater than the max_iou found so far
                            max_iou = iou  # Update max_iou with the current IOU
                            assigned_class = yolo_result["classes"][idx]  # Assign the class of the YOLO bounding box with the highest IOU
                    
                    if max_iou > 0.7:  # If the max_iou is greater than the threshold (0.7 in this case)
                        detection_result["yolo_classes"].append(assigned_class)  # Append the assigned class to the yolo_classes list
                    else:  # If no YOLO bounding box has an IOU greater than the threshold
                        detection_result["yolo_classes"].append(None)  # Append None to the yolo_classes list


    #### Overlap Analysis ####
    if verbose: print(f"Detection analysis")
    non_overlapping_detections, overlapping_detections = split_detections_bbx_by_overlap(detection_results)  # Split detections by overlap

    #### PostProcess ####
    if verbose: print(f"Classification")
    if classification_model_path:
        classification_model = torch.load(classification_model_path).to(device)  # Load classification model

        classify_detections_bbx(non_overlapping_detections, classification_model, classification_idx_to_class_name, CLASSIFICATION_BATCH_SIZE, device, verbose=verbose)  # Classify non-overlapping detections
        classify_detections_bbx(overlapping_detections, classification_model, classification_idx_to_class_name, CLASSIFICATION_BATCH_SIZE, device, verbose=verbose)  # Classify overlapping detections

    #### recombine overlapping and none overlapping, with classes and ordered ####
    detection_results = combine_detections(non_overlapping_detections, overlapping_detections)

    return non_overlapping_detections, overlapping_detections, detection_results  # Return detections and results

def gcd_three(a, b, c):
    return math.gcd(math.gcd(a, b), c)

def detection_data_process_by_video_path(
        video_path, save_outputs_folder: str,
        trex2_api_tokens: list,
        yolo_model_path: str,

        device: str = "cpu",

        classification_model_path: str = None, 
        classification_idx_to_class_name = None,

        CLASSIFICATION_BATCH_SIZE: int = 50,

        reference_images_data: list = None,

        TREX_EVERY_N_FRAMES: float = 24,

        YOLO_BATCH_SIZE: int = 400,  # YOLO batch size for processing frames
        YOLO_EVERY_N_FRAMES: float = 24,  # Number of frames to skip before running YOLO again
        
        USE_BGS: bool = True,
        BG_N_SAMPLES: int = 480,  # Number of images to keep in the queue
        ADD_BG_EVERY: float = 10,
        UPDATE_BG_EVERY: float = 120,  # Frequency to update background

        verbose: bool = False
    ):
    """
    Process a video using the T-Rex pipeline.

    Parameters:
        video_path (str): Path to the input video.
        save_outputs_folder (str): Directory to save the output results.
        trex2_api_tokens (list): List of API tokens for T-Rex.
        reference_images_data (list): list which contains data about each reference image (dict with the keys "image_path" and "boxes" which represents the bounding boxes of the images).
        yolo_model_path (str): Path to the YOLO model.
        device (str): Device to run the models on ('cpu' or 'cuda').
        TREX_EVERY_N_FRAMES (float): Number of frames to process using T-Rex. Can be a float between 0 and 1 indicating the percentage of frames.
        YOLO_BATCH_SIZE (int): Batch size for processing frames using YOLO.
        YOLO_EVERY_N_FRAMES (float): Number of frames to skip before running YOLO again. Can be a float between 0 and 1 indicating the percentage of frames.
        USE_BGS (bool): Whether to use background subtraction.
        BG_N_SAMPLES (int): Number of images to keep in the background queue.
        ADD_BG_EVERY (float): Frequency to add background samples. Can be a float between 0 and 1 indicating the percentage of frames.
        UPDATE_BG_EVERY (float): Frequency to update the background. Can be a float between 0 and 1 indicating the percentage of frames.
        verbose (bool): Whether to print verbose output.

    Returns:
        tuple: Non-overlapping detections, overlapping detections, and all detection results.
    """
    video_frames_folder = os.path.join(save_outputs_folder, 'frames')  # Path to save processed frames
    os.makedirs(video_frames_folder, exist_ok=True)  # Create directory if it doesn't exist

    # Calculate parameters based on video length if between 0 and 1
    if 0 < TREX_EVERY_N_FRAMES <= 1:
        TREX_EVERY_N_FRAMES = len(image_paths) * TREX_EVERY_N_FRAMES  # Adjust T-Rex frame frequency
    
    if 0 < YOLO_EVERY_N_FRAMES <= 1:
        YOLO_EVERY_N_FRAMES = len(image_paths) * YOLO_EVERY_N_FRAMES  # Adjust YOLO frame frequency
    
    if 0 < ADD_BG_EVERY <= 1:
        ADD_BG_EVERY = len(image_paths) * ADD_BG_EVERY  # Adjust background sample addition frequency
    
    if 0 < UPDATE_BG_EVERY <= 1:
        UPDATE_BG_EVERY = len(image_paths) * UPDATE_BG_EVERY  # Adjust background update frequency


    save_to_disk_every_n_frames = gcd_three(TREX_EVERY_N_FRAMES, YOLO_EVERY_N_FRAMES, int(ADD_BG_EVERY)) if USE_BGS else math.gcd(TREX_EVERY_N_FRAMES, YOLO_EVERY_N_FRAMES) # get number of frames to skip
    image_paths = save_video_frames_to_disk(video_path, video_frames_folder, save_to_disk_every_n_frames, verbose=verbose)  # Save video frames to disk
    # image_paths = sorted(image_paths, key=lambda filename: int(filename.rsplit('_', 1)[-1].split('.', 1)[0]))  # Sort image paths by frame number

    # Call the image paths pipeline with the adjusted parameters
    non_overlapping_detections, overlapping_detections, detection_results = detection_data_process_by_image_paths(
        image_paths, save_outputs_folder, trex2_api_tokens, yolo_model_path, device,
        classification_model_path=classification_model_path,
        classification_idx_to_class_name=classification_idx_to_class_name,
        CLASSIFICATION_BATCH_SIZE=CLASSIFICATION_BATCH_SIZE,
        TREX_EVERY_N_FRAMES=int(TREX_EVERY_N_FRAMES),
        YOLO_BATCH_SIZE=YOLO_BATCH_SIZE,
        YOLO_EVERY_N_FRAMES=int(YOLO_EVERY_N_FRAMES),
        reference_images_data=reference_images_data,
        USE_BGS=USE_BGS,
        BG_N_SAMPLES=BG_N_SAMPLES,
        ADD_BG_EVERY=int(ADD_BG_EVERY),
        UPDATE_BG_EVERY=int(UPDATE_BG_EVERY),
        verbose=verbose
    )

    return non_overlapping_detections, overlapping_detections, detection_results  # Return detections and results
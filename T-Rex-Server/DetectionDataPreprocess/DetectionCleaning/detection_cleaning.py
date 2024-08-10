import torch
import numpy as np
from typing import Tuple, Union
import cv2
from PIL import Image
import os
import pickle
from tqdm import tqdm

import torch
start_cuda = torch.cuda.Event(enable_timing=True, blocking=True)
finish_cuda = torch.cuda.Event(enable_timing=True, blocking=True)

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


class RunningImageQueue:
    def __init__(self, T: int, image_shape: Tuple[int, int], use_torch: bool = False):
        """
        Initialize the running image queue.

        Parameters:
        - T (int): The number of images to keep in the queue.
        - image_shape (Tuple[int, int]): The shape of the input images (height, width).
        - use_torch (bool): Boolean flag to use PyTorch tensors instead of NumPy arrays.
        """
        self.T = T
        self.image_shape = image_shape
        self.use_torch = use_torch
        if use_torch:
            self.buffer = torch.zeros((T, *image_shape))
        else:
            self.buffer = np.zeros((T, *image_shape))
        self.index = 0
        self.full = False

    def add_image(self, image: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Add a new image to the queue.

        Parameters:
        - image (Union[np.ndarray, torch.Tensor]): A numpy array or torch tensor representing the new image (shape should match image_shape).
        """
        if self.use_torch:
            if not isinstance(image, torch.Tensor):
                raise ValueError("Image should be a torch.Tensor when use_torch is True")
            if image.shape != self.image_shape:
                raise ValueError(f"Image shape {image.shape} does not match expected shape {self.image_shape}")
            self.buffer[self.index] = image
        else:
            if not isinstance(image, np.ndarray):
                raise ValueError("Image should be a numpy array when use_torch is False")
            if image.shape != self.image_shape:
                raise ValueError(f"Image shape {image.shape} does not match expected shape {self.image_shape}")
            self.buffer[self.index] = torch.from_numpy(image)

        self.index = (self.index + 1) % self.T
        if self.index == 0:
            self.full = True

    def get_running_images(self) -> Union[np.ndarray, torch.Tensor]:
        """
        Get the running images as a numpy array or torch tensor of shape (T, H, W).

        Returns:
        - A numpy array or torch tensor of shape (T, H, W) containing the images in the queue.
        """
        if self.full:
            if self.use_torch:
                return torch.roll(self.buffer, shifts=-self.index, dims=0)
            else:
                return np.roll(self.buffer, shift=-self.index, axis=0)
        else:
            if self.use_torch:
                return self.buffer[:self.index]
            else:
                return self.buffer[:self.index]


def create_BG_from_torch_batch(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Create a background image from a batch of images using the median.

    Parameters:
    - input_tensor (torch.Tensor): A tensor of shape (T, C, H, W) representing a batch of images.

    Returns:
    - torch.Tensor: A tensor of shape (1, C, H, W) representing the background image.
    """
    BG_tensor = input_tensor.median(0, True)[0]
    return BG_tensor

def RGB2BW(input_image):
    if len(input_image.shape) == 2:
        return input_image

    if len(input_image.shape) == 3:
        if type(input_image) == torch.Tensor and input_image.shape[0] == 3:
            grayscale_image = 0.299 * input_image[0:1, :, :] + 0.587 * input_image[1:2, :, :] + 0.114 * input_image[2:3, :, :]
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 3:
            grayscale_image = 0.299 * input_image[:, :, 0:1] + 0.587 * input_image[:, :, 1:2] + 0.114 * input_image[:, :, 2:3]
        else:
            grayscale_image = input_image

    elif len(input_image.shape) == 4:
        if type(input_image) == torch.Tensor and input_image.shape[1] == 3:
            grayscale_image = 0.299 * input_image[:, 0:1, :, :] + 0.587 * input_image[:, 1:2, :, :] + 0.114 * input_image[:, 2:3, :, :]
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 3:
            grayscale_image = 0.299 * input_image[:, :, :, 0:1] + 0.587 * input_image[:, :, :, 1:2] + 0.114 * input_image[:, :, :, 2:3]
        else:
            grayscale_image = input_image

    elif len(input_image.shape) == 5:
        if type(input_image) == torch.Tensor and input_image.shape[2] == 3:
            grayscale_image = 0.299 * input_image[:, :, 0:1, :, :] + 0.587 * input_image[:, :, 1:2, :, :] + 0.114 * input_image[:, :, 2:3, :, :]
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 3:
            grayscale_image = 0.299 * input_image[:, :, :, :, 0:1] + 0.587 * input_image[:, :, :, :, 1:2] + 0.114 * input_image[:, :, :, :, 2:3]
        else:
            grayscale_image = input_image

    return grayscale_image

def get_BGS_from_BG(input_tensor: torch.Tensor, BG_tensor: torch.Tensor, flag_to_uint8: bool, outliers_threshold: int = 30) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the background-subtracted image and outliers.

    Parameters:
    - input_tensor (torch.Tensor): A tensor of shape (T, C, H, W) representing a batch of images.
    - BG_tensor (torch.Tensor): A tensor of shape (1, C, H, W) representing the background image.
    - flag_to_uint8 (bool): Flag to convert the background image to uint8.
    - outliers_threshold (int): Threshold to determine outliers.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
        - BGS_tensor: A tensor representing the background-subtracted image.
        - outliers_tensor: A tensor indicating the outliers.
    """
    BGS_tensor = (input_tensor - BG_tensor).abs().clip(0, 255)  # Get average over channels
    BGS_tensor = RGB2BW(BGS_tensor)
    if flag_to_uint8:
        BGS_tensor = BGS_tensor.type(torch.uint8)
    outliers_tensor = (BGS_tensor > outliers_threshold).float()
    return BGS_tensor, outliers_tensor

def process_image_paths_to_bg_dict(image_paths, bg_tensors_folder, ADD_BG_EVERY=10, UPDATE_BG_EVERY=120, BG_N_SAMPLES=480, verbose=False):
    """
    Process a list of image paths, optionally using Background Subtraction (BGS).

    Parameters:
        image_paths (List[str]): List of paths to the images sorted by order.
        bg_tensors_folder (str): Folder to save background tensors.
        ADD_BG_EVERY (int): Interval for adding images to the background. Default is 10.
        UPDATE_BG_EVERY (int): Interval for updating the background. Default is 120.
        BG_N_SAMPLES (int): Number of samples to use for background calculation. Default is 480.
        verbose (bool): Flag indicating whether to print progress and timing information. Default is False.

    Returns:
        Dict[int, tensor]: Dictionary of frame number as key and bg tensor as value.
    """
    bg_tensors = {}  # Initialize dictionary for background tensors
    
    # Determine the shape of the images
    first_image = cv2.imread(image_paths[0])
    img_height, img_width = first_image.shape[:2]  # Get image height and width
    image_shape = (img_height, img_width, 3)  # Define the expected image shape
    
    # Initialize RunningImageQueue
    running_queue = RunningImageQueue(BG_N_SAMPLES, (3, img_height, img_width), use_torch=True)  
    
    path_make_path_if_none_exists(bg_tensors_folder)
    n_image = len(image_paths)

    # Get latest pickle available
    last_pickle_idx = 0
    for idx in range(n_image):
        current_bg_tensor_filename = str(idx) + '.pkl' # pickle file name
        specific_bg_tensor_full_filename = os.path.join(bg_tensors_folder, current_bg_tensor_filename) # pickle file path
        if os.path.exists(specific_bg_tensor_full_filename): # check if exists
            last_pickle_idx = idx # update to latest index
            bg_tensor = pickle_load(specific_bg_tensor_full_filename) # loading tensor
            bg_tensors[idx] = pickle_load(specific_bg_tensor_full_filename) # adding tensor to dict

    start_idx = last_pickle_idx - ADD_BG_EVERY*BG_N_SAMPLES # starting from latest pickle, going back enough to re fill the runnign queue

    if last_pickle_idx + ADD_BG_EVERY*BG_N_SAMPLES>=n_image: # checking if all images were used
        start_idx = last_pickle_idx

    for idx, image_path in tqdm(enumerate(image_paths), total=n_image, disable=not verbose):  # Loop through each image path
        # adding frame to frames collection for future bg processing 
        if idx>=start_idx:
            if idx % ADD_BG_EVERY == 0:
                # Reading Frame
                frame = cv2.imread(image_path)  # Read the frame
                if frame is None:
                    continue  # Skip if frame is not read successfully

                # Ensure the frame has the same shape as the first image
                if frame.shape != image_shape:
                    raise Exception("All images must be of same shape")
                
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)  # Convert frame to torch tensor
                running_queue.add_image(frame_tensor)  # Add frame tensor to running queue

            # processing current bg if needed
            if running_queue.full and idx % UPDATE_BG_EVERY == 0:  # Check if background needs updating
                current_bg_tensor_filename = str(idx) + '.pkl'
                specific_bg_tensor_full_filename = os.path.join(bg_tensors_folder, current_bg_tensor_filename)
                
                # checking if bg was aleady processed
                if not os.path.exists(specific_bg_tensor_full_filename):
                    running_images = running_queue.get_running_images()  # Get running images
                    bg_tensor = create_BG_from_torch_batch(running_images).type(torch.uint8)  # Create background tensor
                    pickle_save(bg_tensor, specific_bg_tensor_full_filename)  # Save background tensor to disk
                bg_tensors[idx] = bg_tensor  # Append to background tensor dictionary

    return bg_tensors  # Return list of T-Rex image prompts and background tensors

def clean_bounding_boxes_BGS(detections_results, bg_tensors):
    """
    Clean bounding boxes based on the BGS results.

    Parameters:
        detections_results (List[Dict[str, Any]]): List of detection results with bounding boxes.
        bg_tensors (Dict[int, torch.Tensor] or List[torch.Tensor]): Dictionary or List of background tensors indexed by frame count.

    Returns:
        List[Dict[str, Any]]: Cleaned detection results with filtered bounding boxes.
    """
    # Loop through each detection result
    for detection_result in detections_results:
        if 'image_path' in detection_result.keys():
            image_path = detection_result["image_path"]  # Get the image path from the result
        elif 'image' in detection_result.keys():
            image_path = detection_result["image"]  # Get the image path from the result
        
        # Load the original frame
        if isinstance(image_path, str):
            frame = cv2.imread(image_path)  # Load image using OpenCV if path is provided
        elif isinstance(image_path, Image.Image):
            frame = np.array(image_path)  # Convert PIL Image to numpy array
        elif isinstance(image_path, torch.Tensor):
            frame = image_path.permute(1, 2, 0).numpy()  # Convert torch tensor to numpy array
        elif isinstance(image_path, np.ndarray):
            frame = image_path  # Use numpy array directly
        
        bg_tensor_key = detection_result["bg_tensor_key"]  # Get the background tensor key
        bg_tensor = bg_tensors[bg_tensor_key].float()  # Get the background tensor and convert to float

        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()  # Convert frame to torch tensor

        # Get BGS results
        outliers_threshold = 50  # Set outliers threshold
        bg_subtracted_image, _ = get_BGS_from_BG(frame_tensor, bg_tensor, True, outliers_threshold)  # Perform background subtraction
        bg_subtracted_image = bg_subtracted_image.squeeze(0).permute(1, 2, 0).type(torch.uint8).numpy()  # Convert the result back to numpy array

        cleaned_bounding_boxes_idx = []  # Initialize list to store indices of valid bounding boxes
        
        # Loop through each bounding box in the result
        for bbox_idx in range(len(detection_result["boxes"])):
            x1, y1, x2, y2 = detection_result["boxes"][bbox_idx]  # Get bounding box coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to integers
            roi = bg_subtracted_image[y1:y2, x1:x2, :]  # Extract the region of interest from the background subtracted image
            mean_val = roi.mean().item()  # Calculate the mean pixel value in the ROI
            
            # Threshold to decide if the bounding box is valid
            if mean_val > 10:  # Example threshold, you might need to adjust this
                cleaned_bounding_boxes_idx.append(bbox_idx)  # Append valid bounding box index
        
        # Filter bounding boxes, scores, and labels based on valid indices (and all other lists)

        for key in detection_result.keys():
            if isinstance(detection_result[key], list) or isinstance(detection_result[key], np.ndarray):
                detection_result[key] = [i for idx, i in enumerate(detection_result[key]) if idx in cleaned_bounding_boxes_idx]  # Filter key list value

    return detections_results  # Return the list of cleaned results
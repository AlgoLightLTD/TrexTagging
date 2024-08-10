from DetectionDataPreprocess.TRex.trex import TRex2APIWrapper
import concurrent.futures
from time import sleep
from random import choice
from tqdm import tqdm
from typing import Dict, Any, Union, List
import numpy as np
import os
import pickle

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

def perform_inference(image_path: str, prompt_list: List[Dict[str, Any]], trex2_api_token: str, idx: int, max_retries: int = 8, verbose: bool = False) -> Dict[str, Union[str, np.ndarray]]:
    """
    Perform inference using the T-Rex API with the provided image and prompts.

    Parameters:
        image_path (str): Path to the image file.
        prompt_list (List[Dict[str, Any]]): List of prompts for inference.
        trex2_api_token (str): API token for accessing the T-Rex API.
        max_retries (int): Maximum number of retries for the API call. Default is 8.
        verbose (bool): If True, print verbose output. Default is False.

    Returns:
        Dict[str, Union[str, np.ndarray]]: Dictionary containing the inference results including scores, labels, and boxes.
    """
    # Check if the prompt list is empty
    if len(prompt_list) == 0:  # If no prompts are given
        return {
            "message": "No visual prompts were given",  # Return a message indicating no prompts
            "scores": [],  # Empty scores list
            "labels": [],  # Empty labels list
            "boxes": [],  # Empty boxes list
            "idx": idx
        }
    
    # Initialize the T-Rex API wrapper
    trex2 = TRex2APIWrapper(trex2_api_token)  # Create an instance of the T-Rex API wrapper with the provided token

    n_tries = 0  # Initialize retry counter
    while n_tries < max_retries:  # Retry loop for API calls
        try:
            result = trex2.generic_inference(image_path, prompt_list)  # Perform inference with the T-Rex API
            n_tries = max_retries  # Set retries to max to exit loop on success
        except Exception as e:  # Catch any exceptions
            if verbose:  # If verbose mode is enabled
                print(f"Error with T-Rex: {e}")  # Print the error message
            n_tries += 1  # Increment retry counter
            sleep(4)  # Wait for 4 seconds before retrying
            if n_tries == max_retries:  # If maximum retries reached
                return {
                    "message": f"Error with T-Rex: {e} (n_tries == max_retries)",  # Return the error message
                    "scores": [],  # Empty scores list
                    "labels": [],  # Empty labels list
                    "boxes": [],  # Empty boxes list
                    "idx": idx
                }

    # Process the results
    scores = np.array(result["scores"])  # Convert scores to a numpy array
    labels = np.array(result["labels"])  # Convert labels to a numpy array
    boxes = np.array(result["boxes"])  # Convert boxes to a numpy array
    
    return {
        "message": "",  # Empty message on success
        "scores": scores,  # Return scores
        "labels": labels,  # Return labels
        "boxes": boxes,  # Return boxes
        "idx": idx
    }

def perform_inference_with_save(image_path: str, prompt_list: List[Dict[str, Any]], trex2_api_token: str, idx: int, max_retries: int = 8, result_file_path: str = None, verbose: bool = False):
    """
    Perform inference using the T-Rex API and save the results to a file if specified.

    Parameters:
        image_path (str): Path to the image file.
        prompt_list (List[Dict[str, Any]]): List of prompts for inference.
        trex2_api_token (str): API token for accessing the T-Rex API.
        idx (int): Index for the current inference task.
        max_retries (int): Maximum number of retries for the API call. Default is 8.
        result_file_path (str): Path to save the inference results using pickle. If None, results are not saved.
        verbose (bool): If True, print verbose output. Default is False.

    Returns:
        Dict[str, Union[str, np.ndarray]]: Dictionary containing the inference results including scores, labels, and boxes.
    """
    
    # Nested function to handle loading from file or performing inference
    def load_or_perform_inference():
        # Check if the result file exists
        if os.path.exists(result_file_path):
            trex_result = pickle_load(result_file_path)  # Load the result from the file
            # Check if the loaded result is valid (non-empty message)
            if trex_result and "message" in trex_result and not trex_result["message"]:
                return trex_result
        # Perform inference if the result file does not exist or the result is invalid
        return perform_inference(image_path, prompt_list, trex2_api_token, idx=idx, max_retries=max_retries, verbose=verbose)
    
    # Perform inference or load existing results
    trex_result = load_or_perform_inference()
    
    # Save the result to the file if a result file path is specified and it does not already exist
    if result_file_path and not os.path.exists(result_file_path):
        pickle_save(trex_result, result_file_path)
    
    return trex_result  # Return the inference result

def trex_infer(images_prompts: List[Dict[str, Any]], trex2_api_token_list: List[str], max_retries: int = 8, trex_save_path:str = None, verbose: bool = False) -> List[Dict[str, Union[str, np.ndarray]]]:
    """
    Perform T-Rex inference on a list of images and their corresponding prompts.

    Parameters:
        images_prompts (List[Dict[str, Any]]): List of dictionaries containing image paths and prompts.
        trex2_api_token_list (List[str]): List of API tokens for T-Rex API.
        verbose (bool): If True, print verbose output. Default is False.

    Returns:
        List[Dict[str, Union[str, np.ndarray]]]: List of inference results for each image.
    """
    assert len(trex2_api_token_list), "trex2_api_token_list must contain at list one token"

    results = []  # Initialize list to store results
    with concurrent.futures.ThreadPoolExecutor() as executor:  # Use ThreadPoolExecutor for parallel processing
        future_to_image = {}  # Dictionary to map futures to image paths
        for image_prompts in images_prompts:  # Loop through each image and its prompts
            prompt_list = image_prompts["prompts"]  # Extract the prompt list
            image_path = image_prompts["image_path"]  # Extract the image path

            # Get index of current image to inference
            idx = None
            if "idx" in image_prompts:
                idx = image_prompts["idx"]
            
            trex2_api_token = choice(trex2_api_token_list)  # Select a random API token
            
            # create file path for current result
            result_file_path = None
            if trex_save_path:
                result_file_path = os.path.join(trex_save_path, f"trex_{idx}")
            
            # call the inference as future for parallel computing
            future = executor.submit(perform_inference_with_save, image_path, prompt_list, trex2_api_token, idx=idx, max_retries=max_retries, result_file_path=result_file_path, verbose=verbose)  # Submit the inference task to the executor
            future_to_image[future] = image_path  # Map the future to the image path
        
        for future in tqdm(concurrent.futures.as_completed(future_to_image), total=len(future_to_image), disable=not verbose):  # Loop through completed futures
            image_path = future_to_image[future]  # Get the image path for the current future
            try:
                result = future.result()  # Get the result from the future
                results.append({"image_path": image_path} | result)  # Append the result to the results list
                if verbose:  # If verbose mode is enabled
                    print(f"Inference result for {image_path}: {result}")  # Print the inference result
            except Exception as exc:  # Catch any exceptions
                print(f'Image {image_path} generated an exception: {exc}')  # Print the exception message

    return results  # Return the list of results
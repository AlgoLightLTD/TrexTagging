import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

def classify_detection_bbx(image, bounding_boxes, model, idx_to_class, batch_size=1, resize_dim=(224, 224), device: str ='cpu'):
    """
    Classify objects within bounding boxes in an image using a given model.

    Parameters:
        image (str, np.ndarray, or torch.Tensor): Path to the image file, the image as a NumPy array, or a torch tensor.
        bounding_boxes (List[Tuple[int, int, int, int]]): List of bounding boxes (x1, y1, x2, y2).
        model (torch.nn.Module): PyTorch model for classification.
        idx_to_class (Dict[int, str] or List[str]): Dictionary mapping class indices to class names.
        batch_size (int): Number of images to process in a batch.
        resize_dim (Tuple[int, int]): Dimensions to resize the bounding boxes.

    Returns:
        Tuple[List[str], List[float]]: Two lists containing class names and confidence scores for each bounding box.
    """
    class_names = []  # Initialize list to store class names
    confidence_scores = []  # Initialize list to store confidence scores
    batched_rois = []  # Initialize list to store batched regions of interest (ROIs)

    model = model.to(device)

    def predict_batch():
        """
        Perform prediction on the batched ROIs and update class names and confidence scores.
        """
        rois_tensor = torch.stack(batched_rois).to(device)  # Stack ROIs into a single tensor
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(rois_tensor)  # Perform inference using the model
            _, predicted = torch.max(outputs.cpu(), 1)  # Get the predicted class indices
            confidences = outputs.cpu()  # Get the raw output scores

        for i in range(len(batched_rois)):  # Loop through each ROI in the batch
            class_names.append(idx_to_class[predicted[i].item()])  # Append the class name
            confidence_scores.append(confidences[i][predicted[i]].item())  # Append the confidence score

    # Load and convert the image
    if isinstance(image, str):  # If image is a file path
        img = Image.open(image).convert("RGB")  # Open and convert to RGB
    elif isinstance(image, np.ndarray):  # If image is a numpy array
        img = Image.fromarray(image.astype('uint8')).convert("RGB")  # Convert to PIL Image and then to RGB
    elif isinstance(image, torch.Tensor):  # If image is a torch tensor
        img = transforms.ToPILImage()(image).convert("RGB")  # Convert to PIL Image and then to RGB
    else:
        raise TypeError("image should be either a file path, a NumPy array, or a torch tensor")  # Raise error if type is unsupported

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize(resize_dim),  # Resize the image
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the tensor
    ])

    # Loop through each bounding box
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox  # Unpack bounding box coordinates
        roi = img.crop((x1, y1, x2, y2))  # Crop the region of interest
        roi = transform(roi)  # Apply transformations
        batched_rois.append(roi)  # Add the transformed ROI to the batch

        if len(batched_rois) == batch_size:  # If the batch is full
            predict_batch()  # Perform prediction
            batched_rois = []  # Reset the batch

    # Process any remaining ROIs
    if len(batched_rois) > 0:  # If there are remaining ROIs
        predict_batch()  # Perform prediction on the remaining ROIs

    return class_names, confidence_scores  # Return the class names and confidence scores

def classify_detections_bbx(detection_results, classification_model, idx_to_class, classification_model_batch_size=10, device='cpu', verbose=False):
    """
    Classify objects within bounding boxes in detection results using a given classification model.

    Parameters:
        detection_results (List[Dict[str, Any]]): List of detection results, each containing an image path and bounding boxes.
        classification_model (torch.nn.Module): PyTorch model for classification.
        idx_to_class (Dict[int, str] or List[str]): Dictionary or list mapping class indices to class names.
        classification_model_batch_size (int): Number of images to process in a batch. Default is 10.
        device (str): Device to run the model on ('cpu' or 'cuda'). Default is 'cpu'.
        verbose (bool): If True, display a progress bar using tqdm. Default is False.

    Returns:
        None: The function modifies the input detection_results in place, adding class names and confidence scores.
    """
    # Use tqdm progress bar if verbose is True
    iterator = tqdm(detection_results, desc="Classifying detections", disable=not verbose)
    
    # Loop through each detection result
    for detection_result in iterator:
        # Classify the bounding boxes in the current detection result
        classes, confidence = classify_detection_bbx(
            detection_result["image_path"],  # Path to the image
            detection_result['boxes'],  # List of bounding boxes
            classification_model,  # Classification model
            idx_to_class,  # Mapping from class indices to class names
            classification_model_batch_size,  # Batch size for classification
            device=device  # Device to run the model on
        )
        # Add the classification results to the detection result
        detection_result["classes"] = classes  # Add class names to the detection result
        detection_result["classes_confidence"] = confidence  # Add confidence scores to the detection result
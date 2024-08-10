from typing import Dict, List, Tuple
import numpy as np

def calculate_iou(box1, box2):
        """
        Calculate Intersection Over Union (IOU) of two bounding boxes.

        Args:
            box1 (List[int]): First bounding box, list [x1, y1, x2, y2].
            box2 (List[int]): Second bounding box, list [x1, y1, x2, y2].

        Returns:
            float: IOU value.
        """
        x1 = max(box1[0], box2[0])  # Calculate the x-coordinate of the intersection top-left corner
        y1 = max(box1[1], box2[1])  # Calculate the y-coordinate of the intersection top-left corner
        x2 = min(box1[2], box2[2])  # Calculate the x-coordinate of the intersection bottom-right corner
        y2 = min(box1[3], box2[3])  # Calculate the y-coordinate of the intersection bottom-right corner
        inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)  # Calculate the area of the intersection rectangle
        
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)  # Calculate the area of the first bounding box
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)  # Calculate the area of the second bounding box
        
        iou = inter_area / float(box1_area + box2_area - inter_area)  # Calculate the IOU by dividing the intersection area by the union area
        return iou

def split_detection_bbx_by_overlap(detection_result):
    """
    Separate detection result of a single image into two detection results,
    one for bounding boxes that do not overlap other bounding boxes and another for overlapping ones.

    Parameters:
        detection_result (Dict[str, Any]): Detection result with bounding boxes.

    Returns:
        Dict[str, Any], Dict[str, Any]: Detection results separated into non-overlapping and overlapping.
    """

    non_overlapping_detections = {key: [] for key in detection_result.keys()}  # Initialize dictionary for non-overlapping detections
    overlapping_detections = {key: [] for key in detection_result.keys()}  # Initialize dictionary for overlapping detections

    boxes = detection_result["boxes"]  # Get the list of bounding boxes from the detection result
    overlap_flags = [False] * len(boxes)  # Initialize a list to track which boxes are overlapping

    # Check each pair of bounding boxes for overlap
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if calculate_iou(boxes[i], boxes[j]) > 0:  # If the IOU is greater than 0, the boxes overlap
                overlap_flags[i] = True  # Mark the first box as overlapping
                overlap_flags[j] = True  # Mark the second box as overlapping

    # Separate the detections into non-overlapping and overlapping based on the flags
    for idx, flag in enumerate(overlap_flags):
        if flag:
            for key in detection_result:
                if isinstance(detection_result[key], list) or isinstance(detection_result[key], np.ndarray):
                    if idx < len(detection_result[key]):
                        overlapping_detections[key].append(detection_result[key][idx])  # Copy values to overlapping detections
                else:
                    overlapping_detections[key] = detection_result[key]
        else:
            for key in detection_result:
                if isinstance(detection_result[key], list) or isinstance(detection_result[key], np.ndarray):
                    if idx < len(detection_result[key]):
                        non_overlapping_detections[key].append(detection_result[key][idx])  # Copy values to non-overlapping detections
                else:
                    non_overlapping_detections[key] = detection_result[key]

    return non_overlapping_detections, overlapping_detections  # Return the separated detection results

def split_detections_bbx_by_overlap(detection_results):
    """
    Separate detection results of a images into two detection results,
    one for bounding boxes that do not overlap other bounding boxes and another for overlapping ones.

    Parameters:
        detection_result List[Dict[str, Any]]: List of Detection results with bounding boxes.

    Returns:
        List[Dict[str, Any]], List[Dict[str, Any]]: 2 lists with Detection results separated into non-overlapping and overlapping.
    """
    
    non_overlapping_detections = []
    overlapping_detections = []
    for detection_result in detection_results:
        non_overlapping_detection, overlapping_detection = split_detection_bbx_by_overlap(detection_result)
        if len(non_overlapping_detection["image_path"])>0:
            non_overlapping_detections.append(non_overlapping_detection)
        if len(overlapping_detection["image_path"])>0:
            overlapping_detections.append(overlapping_detection)

    return non_overlapping_detections, overlapping_detections  # Return the separated detection results
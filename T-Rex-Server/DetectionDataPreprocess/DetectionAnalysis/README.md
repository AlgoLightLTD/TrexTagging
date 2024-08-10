
# Detection-Analysis

This repository contains scripts for analyzing and processing detection results, specifically focusing on handling overlapping and non-overlapping bounding boxes in object detection.

## File Overview

### detection_analysis.py

This script includes the following functions:

1. `split_detection_bbx_by_overlap(detection_result)`
    - Separates detection results of a single image into non-overlapping and overlapping bounding boxes.
    - **Parameters:**
      - `detection_result`: Detection result with bounding boxes.
    - **Returns:**
      - Two dictionaries containing detection results separated into non-overlapping and overlapping bounding boxes.

2. `split_detections_bbx_by_overlap(detection_results)`
    - Separates detection results of multiple images into non-overlapping and overlapping bounding boxes.
    - **Parameters:**
      - `detection_results`: List of detection results, each containing bounding boxes.
    - **Returns:**
      - Two lists of dictionaries containing detection results separated into non-overlapping and overlapping bounding boxes.

## Setup

### Prerequisites

- Python 3.x
- numpy

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Detection-Analysis.git
   cd Detection-Analysis
   ```

2. Install the required packages:
   ```
   pip install numpy
   ```

## Usage

### Splitting Bounding Boxes by Overlap

1. Import the required functions:
   ```python
   from detection_analysis import split_detection_bbx_by_overlap, split_detections_bbx_by_overlap
   ```

2. Prepare your detection results:
   ```python
   detection_result = {
       "boxes": [(50, 50, 150, 150), (100, 100, 200, 200)],
       "scores": [0.9, 0.75],
       "labels": [1, 2]
   }
   ```

3. Split bounding boxes by overlap for a single image:
   ```python
   non_overlapping, overlapping = split_detection_bbx_by_overlap(detection_result)
   print(non_overlapping, overlapping)
   ```

4. Split bounding boxes by overlap for multiple images:
   ```python
   detection_results = [
       {
           "boxes": [(50, 50, 150, 150), (100, 100, 200, 200)],
           "scores": [0.9, 0.75],
           "labels": [1, 2]
       },
       ...
   ]

   non_overlapping_list, overlapping_list = split_detections_bbx_by_overlap(detection_results)
   print(non_overlapping_list, overlapping_list)
   ```

## License

This project is licensed under the MIT License. See the LICENSE file for details.


# T-Rex

This repository contains the T-Rex pipeline for object detection and classification. It integrates various components from other repositories to provide a comprehensive solution for processing videos and images using the T-Rex model.

## File Overview

### TRex.py

This script includes functions for performing inference using the T-Rex API:
- `pickle_save(obj, file_path)`
    - Saves an object to a file using pickle.
    - **Parameters:**
      - `obj`: The object to be pickled.
      - `file_path`: The file path where the object will be saved.

- `pickle_load(file_path)`
    - Loads an object from a pickle file.
    - **Parameters:**
      - `file_path`: The file path from where the object will be loaded.
    - **Returns:**
      - The object that was loaded from the file.

- `perform_inference(image_path, prompt_list, trex2_api_token, idx, max_retries=8, verbose=False)`
    - Performs inference using the T-Rex API.
    - **Parameters:**
      - `image_path`: Path to the image file.
      - `prompt_list`: List of prompts for inference.
      - `trex2_api_token`: API token for accessing the T-Rex API.
      - `idx`: Index for the current inference task.
      - `max_retries`: Maximum number of retries for the API call.
      - `verbose`: If True, prints verbose output.
    - **Returns:**
      - Dictionary containing the inference results including scores, labels, and boxes.

## Setup

### Prerequisites

- Python 3.x
- torch
- numpy
- opencv-python
- pillow
- tqdm
- ultralytics

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/T-Rex.git
   cd T-Rex
   ```

2. Install the required packages:
   ```
   pip install torch numpy opencv-python pillow tqdm ultralytics
   ```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

import io
import cv2
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import random
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-30, 30), shear=(-15, 15), p=0.5),
    A.Equalize(p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.MotionBlur(blur_limit=7, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

class ImageDataset(Dataset):
    """
    Custom Dataset class for loading and processing images.
    
    Parameters:
    - data_dirs: List of directories containing image data.
    - transform: Transformations to apply to images.
    - use_partial_dataset: Flag to use a partial dataset.
    - dataset_percentage: Percentage of the dataset to use if use_partial_dataset is True.
    - balance_dataset: Flag to balance the dataset by class.
    """
    def __init__(self, data_dirs, transform=None, use_partial_dataset=False, dataset_percentage=0.5, balance_dataset: bool = True):
        self.data_dirs = data_dirs  # Directories containing the image data
        self.transform = transform  # Transformations to apply to images
        self.samples = []  # List to store image paths and labels
        self.samples_by_label = {}  # Dictionary to store samples by class

        self.image_size_distribution_graph = None  # Graph for image size distribution
        self.class_distribution_graph = None  # Graph for class distribution
        self.balanced_class_distribution_graph = None  # Graph for balanced class distribution

        # Iterate through data directories and collect image paths and labels
        for data_dir in self.data_dirs:  # Loop over each data directory
            for class_name in os.listdir(data_dir):  # Loop over each class in the directory
                class_dir = os.path.join(data_dir, class_name)  # Create full path for the class
                if os.path.isdir(class_dir):  # Check if the path is a directory
                    for image_path in glob(os.path.join(class_dir, '**', '*'), recursive=True):  # Recursively get all image paths
                        if image_path.lower().endswith(('.png', '.jpg', 'jpeg')):  # Filter image files by extension
                            self.samples.append((image_path, class_name))  # Add image path and class to samples list

                            if class_name not in self.samples_by_label:  # Initialize list for new class
                                self.samples_by_label[class_name] = []
                            self.samples_by_label[class_name].append((image_path, class_name))  # Add image path and class to dictionary

        #### dataset analysis ####
        self.image_size_distribution_graph = self.get_image_size_disribution_graph()  # Generate graph for image sizes
        self.class_distribution_graph = self.get_class_distribution_graph()  # Generate graph for class distribution

        # Balance class distribution by matching to the class with maximum samples
        if balance_dataset:  # Check if dataset balancing is enabled
            self.balance_dataset()  # Balance the dataset
            self.balanced_class_distribution_graph = self.get_class_distribution_graph()  # Generate graph for balanced class distribution
        
        if use_partial_dataset:  # Check if using a partial dataset
            self.set_partial(dataset_percentage)  # Set partial dataset size

        # Create label to index mapping
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(set(label for _, label in self.samples)))}  # Mapping from label to index
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}  # Mapping from index to label

    def __len__(self):
        return len(self.samples)  # Return the number of samples

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]  # Get image path and label by index
        img = cv2.imread(file_path)  # Read the image using OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB
        label_idx = self.label_to_index[label]  # Get the label index

        if self.transform:  # Apply transformations if specified
            augmented = self.transform(image=img)  # Apply the transformations
            img = augmented['image']  # Extract the transformed image

        return img, label_idx  # Return the transformed image and label index
    
    def get_image_size_disribution_graph(self):
        """
        Generate a histogram of image sizes.
        
        Returns:
        - PIL Image: Histogram of image sizes.
        """
        image_sizes = []  # List to store image sizes
        for image_path, _ in self.samples:  # Loop over each sample
            with Image.open(image_path) as img:  # Open image using PIL
                width, height = img.size  # Get image dimensions
                image_sizes.append((width, height))  # Add dimensions to list
        
        image_sizes = np.array(image_sizes)  # Convert to numpy array for easier manipulation
        widths = image_sizes[:, 0]  # Extract widths
        heights = image_sizes[:, 1]  # Extract heights
        
        min_height = heights.min()  # Calculate the minimum height
        min_height_width = widths[heights.argmin()]  # Corresponding width for the minimum height
        min_width = widths.min()  # Calculate the minimum width
        min_width_height = heights[widths.argmin()]  # Corresponding height for the minimum width
        
        plt.figure(figsize=(12, 6))  # Set figure size

        plt.hist(widths, bins=30, alpha=0.5, label='Width', color='blue')  # Create histogram for widths
        plt.hist(heights, bins=30, alpha=0.5, label='Height', color='red')  # Create histogram for heights
        
        plt.title(f'Histogram of Image Sizes (Widths and Heights)\n(min height, with width): ({min_height}, {min_height_width}) (min width, with height): ({min_width}, {min_width_height})')  # Add title
        plt.xlabel('Size (pixels)')  # Add x-axis label
        plt.ylabel('Frequency')  # Add y-axis label
        plt.legend(loc='upper right')  # Add legend

        buf = io.BytesIO()  # Create a bytes buffer
        plt.savefig(buf, format='PNG')  # Save the plot to the buffer
        plt.close()  # Close the plot
        buf.seek(0)  # Move to the start of the buffer

        image = Image.open(buf)  # Create a PIL image from the buffer

        return image  # Return the image
    
    def get_class_distribution_graph(self):
        """
        Generate a bar graph of class distribution.
        
        Returns:
        - PIL Image: Bar graph of class distribution.
        """
        class_counts = {}  # Dictionary to store class counts
        for class_name, samples in self.samples_by_label.items():  # Loop over each class
            class_counts[class_name] = len(samples)  # Count samples per class

        classes = list(class_counts.keys())  # Get class names
        n_samples = list(class_counts.values())  # Get number of samples per class

        plt.figure(figsize=(10, 5))  # Set figure size
        plt.bar(classes, n_samples, color='skyblue')  # Create bar graph

        plt.title('Bar Graph of Categories')  # Add title
        plt.xlabel('Classes')  # Add x-axis label
        plt.ylabel('N Samples')  # Add y-axis label

        buf = io.BytesIO()  # Create a bytes buffer
        plt.savefig(buf, format='PNG')  # Save the plot to the buffer
        plt.close()  # Close the plot
        buf.seek(0)  # Move to the start of the buffer

        image = Image.open(buf)  # Create a PIL image from the buffer

        return image  # Return the image
    
    def balance_dataset(self):
        """
        Balance the dataset by matching to the class with the maximum samples.
        """
        max_samples = max([len(samples) for samples in self.samples_by_label.values()])  # Find maximum number of samples in any class
        balanced_samples = []  # List to store balanced samples
        for key, samples in self.samples_by_label.items():  # Loop over each class
            if len(samples) < max_samples:  # Check if the class has fewer samples than the maximum
                self.samples_by_label[key].extend(samples * (max_samples // len(samples)) + random.sample(samples, max_samples % len(samples)))  # Balance the class samples
            balanced_samples.extend(self.samples_by_label[key])  # Add samples to balanced list
        self.samples = balanced_samples  # Update samples with balanced samples
    
    def set_partial(self, dataset_percentage):
        """
        Reduce the dataset size by a specified percentage.
        
        Parameters:
        - dataset_percentage: Percentage of the dataset to retain.
        """
        print(f"Using Partial Dataset ({dataset_percentage*100:.2f}%)")  # Print the percentage of the dataset being used
        reduced_samples = []  # List to store reduced samples
        for _, samples in self.samples_by_label.items():  # Loop over each class
            reduced_sample_count = int(len(samples) * dataset_percentage)  # Calculate reduced sample count
            reduced_samples.extend(random.sample(samples, reduced_sample_count))  # Add reduced samples to list
        self.samples = reduced_samples  # Update samples with reduced samples

    def save_dataset(self, json_path):
        """
        Save the dataset information to a JSON file.
        
        Parameters:
        - json_path: Path to the JSON file.
        """
        dataset_info = {
            'data_dirs': self.data_dirs,  # Data directories
            'samples': self.samples,  # Samples
            'samples_by_label': self.samples_by_label,  # Samples by label
            'label_to_index': self.label_to_index,  # Label to index mapping
            'index_to_label': self.index_to_label  # Index to label mapping
        }
        with open(json_path, 'w') as f:  # Open the JSON file for writing
            json.dump(dataset_info, f)  # Save dataset information

    def load_dataset(self, json_path):
        """
        Load the dataset information from a JSON file.
        
        Parameters:
        - json_path: Path to the JSON file.
        """
        with open(json_path, 'r') as f:  # Open the JSON file for reading
            dataset_info = json.load(f)  # Load dataset information
        
        self.data_dirs = dataset_info['data_dirs']  # Update data directories
        self.samples = dataset_info['samples']  # Update samples
        self.samples_by_label = dataset_info['samples_by_label']  # Update samples by label
        self.label_to_index = dataset_info['label_to_index']  # Update label to index mapping
        self.index_to_label = dataset_info['index_to_label']  # Update index to label mapping

def set_memory_limits(device_ids, memory_fractions):
    """
    Set memory limits for specific GPU devices.
    
    Parameters:
    - device_ids: List of GPU device IDs.
    - memory_fractions: List of memory fractions for each GPU.
    """
    for device_id, memory_fraction in zip(device_ids, memory_fractions):  # Loop over each device and memory fraction
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device_id)  # Set memory limit for the device

def save_split_indices(indices, json_path):
    """
    Save split indices to a JSON file.
    
    Parameters:
    - indices: List or tensor of indices.
    - json_path: Path to the JSON file.
    """
    indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else indices  # Convert tensor to list if necessary
    with open(json_path, 'w') as f:  # Open the JSON file for writing
        json.dump(indices_list, f)  # Save indices to the JSON file

def load_split_indices(json_path):
    """
    Load split indices from a JSON file.
    
    Parameters:
    - json_path: Path to the JSON file.
    
    Returns:
    - indices: List of indices.
    """
    with open(json_path, 'r') as f:  # Open the JSON file for reading
        indices = json.load(f)  # Load indices from the JSON file
    return indices  # Return the indices

def compute_class_weights(class_counts):
    """
    Compute class weights for balancing.
    
    Parameters:
    - class_counts: Dictionary of class counts.
    
    Returns:
    - Tensor: Tensor of class weights.
    """
    total_samples = sum(class_counts.values())  # Calculate the total number of samples
    class_weights = {label: total_samples / (len(class_counts) * count) for label, count in class_counts.items()}  # Compute class weights
    weights = [class_weights[label] for label in sorted(class_counts.keys())]  # Create a list of weights
    return torch.tensor(weights)  # Convert the list to a tensor

def plot_metrics(metrics, labels, title, save_path):
    """
    Plot and save bar graph of metrics.
    
    Parameters:
    - metrics: List of metric values.
    - labels: List of metric labels.
    - title: Title of the plot.
    - save_path: Path to save the plot.
    """
    plt.figure(figsize=(10, 7))  # Set figure size
    x = np.arange(len(labels))  # Create an array of label indices
    plt.bar(x, metrics)  # Create a bar graph of metrics
    plt.xticks(x, labels)  # Set x-axis labels
    plt.title(title)  # Set plot title
    plt.xlabel('Metrics')  # Set x-axis label
    plt.ylabel('Value')  # Set y-axis label
    for i, v in enumerate(metrics):  # Loop over each metric value
        plt.text(x=i, y=v + 0.01, s=f"{v:.4f}", ha='center')  # Add text above each bar
    plt.savefig(save_path)  # Save the plot
    plt.close()  # Close the plot

def plot_confusion_matrix(true_labels, pred_labels, class_names, title='Confusion Matrix', save_path=None):
    """
    Plot and save confusion matrix.
    
    Parameters:
    - true_labels: List of true labels.
    - pred_labels: List of predicted labels.
    - class_names: List of class names.
    - title: Title of the plot.
    - save_path: Path to save the plot (optional).
    """
    cm = confusion_matrix(true_labels, pred_labels)  # Compute confusion matrix
    plt.figure(figsize=(10, 7))  # Set figure size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)  # Plot confusion matrix
    plt.xlabel('Predicted')  # Set x-axis label
    plt.ylabel('True')  # Set y-axis label
    plt.title(title)  # Set plot title
    if save_path:  # Check if save path is provided
        plt.savefig(save_path)  # Save the plot
    plt.show()  # Show the plot
    plt.close()  # Close the plot

def load_dataset(images_folder_paths: list, transform, TRAIN_BATCH_SIZE: int, TEST_BATCH_SIZE: int, balance_dataset: bool = True, use_partial_dataset: bool = False, dataset_percentage: float = 1.0):
    """
    Load dataset and create DataLoader objects for training and validation.
    
    Parameters:
    - images_folder_paths: List of image folder paths.
    - transform: Transformations to apply to images.
    - TRAIN_BATCH_SIZE: Batch size for training.
    - TEST_BATCH_SIZE: Batch size for validation.
    - balance_dataset: Flag to balance the dataset by class.
    - use_partial_dataset: Flag to use a partial dataset.
    - dataset_percentage: Percentage of the dataset to use if use_partial_dataset is True.
    
    Returns:
    - dataset: ImageDataset object.
    - train_loader: DataLoader for training.
    - val_loader: DataLoader for validation.
    """
    dataset = ImageDataset(images_folder_paths, transform=transform, use_partial_dataset=use_partial_dataset, dataset_percentage=dataset_percentage, balance_dataset=balance_dataset)  # Create the dataset
    
    train_size = int(0.8 * len(dataset))  # Calculate training set size (80% of dataset)
    val_size = len(dataset) - train_size  # Calculate validation set size (remaining 20%)
    
    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])  # Split dataset into training and validation sets

    train_loader = DataLoader(train_subset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)  # Create DataLoader for training
    val_loader = DataLoader(val_subset, batch_size=TEST_BATCH_SIZE, shuffle=False)  # Create DataLoader for validation
    
    return dataset, train_loader, val_loader  # Return the dataset and DataLoaders
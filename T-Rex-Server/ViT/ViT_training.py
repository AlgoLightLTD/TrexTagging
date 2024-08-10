import os
from .ViT_training_utils import compute_class_weights, load_dataset, plot_metrics, plot_confusion_matrix
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from timm.models.vision_transformer import vit_base_patch16_224
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, accuracy_score

def train_model(model, train_loader, val_loader, criterion, optimizer, writer, class_names, device: str, num_epochs: int = 10, start_from: int = 0, save_folder: str = './', verbose=True):
    """
    Train the vision transformer model.
    
    Parameters:
    - model: The model to train.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - criterion: Loss function.
    - optimizer: Optimizer for training.
    - writer: TensorBoard SummaryWriter for logging.
    - class_names: List of class names.
    - device: Device to run the training on.
    - num_epochs: Number of epochs to train.
    - start_from: Epoch number to start training from.
    - save_folder: Folder to save the model and metrics.
    """
    best_val_accuracy = 0.0  # Initialize best validation accuracy
    batch_len = train_loader.batch_size  # Batch length
    batch_losses = []  # List to store batch losses
    epoch_losses = []  # List to store epoch losses
    best_epoch = 0  # Variable to store the best epoch

    for epoch in range(start_from, num_epochs):  # Loop over epochs
        if verbose: print() 
        model.train()  # Set the model to training mode
        running_loss = 0.0  # Initialize running loss
        epoch_loss = 0  # Initialize epoch loss
        running_correct = 0  # Initialize running correct predictions
        running_total = 0  # Initialize running total samples
        running_accuracy = 0  # Initialize running accuracy
        all_labels = []  # List to store all labels
        all_predictions = []  # List to store all predictions
        n_batches = len(train_loader)  # Number of batches in the training loader
        n_batch = epoch * n_batches  # Batch counter for TensorBoard

        for batch, (inputs, labels) in enumerate(train_loader, start=1):  # Loop over batches
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the device
            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the model

            _, predicted = torch.max(outputs, 1)  # Get predictions
            running_total = labels.size(0)  # Total samples in the batch
            running_correct = (predicted == labels).sum().item()  # Correct predictions in the batch
            all_labels.extend(labels.cpu().numpy())  # Store labels
            all_predictions.extend(predicted.cpu().numpy())  # Store predictions

            n_batch += 1  # Increment batch counter
            epoch_loss += loss.item()  # Accumulate epoch loss
            running_loss = loss.item() / batch_len  # Compute running loss
            writer.add_scalar('Running Loss', running_loss, n_batch)  # Log running loss
            batch_losses.append(running_loss)  # Store running loss
            running_accuracy = running_correct / running_total  # Compute running accuracy
            writer.add_scalar('Running Accuracy', running_accuracy, n_batch)  # Log running accuracy

            if len(all_labels) > 1:  # Compute precision and recall if more than one label
                running_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)  # Compute running precision
                running_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)  # Compute running recall
                writer.add_scalar('Running Precision', running_precision, n_batch)  # Log running precision
                writer.add_scalar('Running Recall', running_recall, n_batch)  # Log running recall
                if verbose: print(f"\rEpoch: {epoch+1}/{num_epochs}, Batch {batch}/{n_batches}, Running Loss: {running_loss}, Running Accuracy: {running_accuracy:.4f}, Running Precision: {running_precision:.4f}, Running Recall: {running_recall:.4f}", end='')  # Print training progress

        epoch_loss /= len(train_loader)  # Compute average epoch loss
        epoch_accuracy = accuracy_score(all_labels, all_predictions)  # Compute epoch accuracy
        epoch_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)  # Compute epoch precision
        epoch_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)  # Compute epoch recall

        writer.add_scalar('Training Loss', epoch_loss, epoch)  # Log training loss
        epoch_losses.append(epoch_loss)  # Store epoch loss
        writer.add_scalar('Training Accuracy', epoch_accuracy, epoch)  # Log training accuracy
        writer.add_scalar('Training Precision', epoch_precision, epoch)  # Log training precision
        writer.add_scalar('Training Recall', epoch_recall, epoch)  # Log training recall

        if verbose: print(f"\nEpoch: {epoch+1}/{num_epochs}, Training Loss: {epoch_loss}, Training Accuracy: {epoch_accuracy:.4f}, Training Precision: {epoch_precision:.4f}, Training Recall: {epoch_recall:.4f}")  # Print training metrics

        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0  # Initialize validation loss
        correct = 0  # Initialize correct predictions
        total = 0  # Initialize total samples
        all_labels = []  # List to store validation labels
        all_predictions = []  # List to store validation predictions

        with torch.no_grad():  # Disable gradient calculation
            for inputs, labels in tqdm(val_loader, disable=not verbose):  # Loop over validation batches
                inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the device
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                val_loss += loss.item()  # Accumulate validation loss

                _, predicted = torch.max(outputs, 1)  # Get predictions
                total += labels.size(0)  # Total samples in the batch
                correct += (predicted == labels).sum().item()  # Correct predictions in the batch
                all_labels.extend(labels.cpu().numpy())  # Store labels
                all_predictions.extend(predicted.cpu().numpy())  # Store predictions

        val_loss /= len(val_loader)  # Compute average validation loss
        val_accuracy = accuracy_score(all_labels, all_predictions)  # Compute validation accuracy
        val_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)  # Compute validation precision
        val_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)  # Compute validation recall

        writer.add_scalar('Validation Loss', val_loss, epoch)  # Log validation loss
        writer.add_scalar('Validation Accuracy', val_accuracy, epoch)  # Log validation accuracy
        writer.add_scalar('Validation Precision', val_precision, epoch)  # Log validation precision
        writer.add_scalar('Validation Recall', val_recall, epoch)  # Log validation recall

        if verbose: print(f"Epoch: {epoch+1}/{num_epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy:.4f}, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}")  # Print validation metrics

        # Save model and metrics each epoch
        epoch_folder = os.path.join(save_folder, f'epoch_{epoch+1}')  # Create folder for the current epoch
        os.makedirs(epoch_folder, exist_ok=True)  # Create the folder if it doesn't exist
        torch.save(model.state_dict(), os.path.join(epoch_folder, "model.pt"))  # Save model state

        train_labels = ['Accuracy', 'Precision', 'Recall']  # Labels for training metrics
        plot_metrics((epoch_accuracy, epoch_precision, epoch_recall), train_labels, 'Training Metrics',
                     os.path.join(epoch_folder, 'training_metrics.png'))  # Plot and save training metrics
        plot_metrics((val_accuracy, val_precision, val_recall), train_labels, 'Validation Metrics',
                     os.path.join(epoch_folder, 'validation_metrics.png'))  # Plot and save validation metrics

        plot_confusion_matrix(all_labels, all_predictions, class_names,
                              title='Confusion Matrix for Validation Set', save_path=os.path.join(epoch_folder, 'confusion_matrix_val.png'))  # Plot and save confusion matrix

        # Check if this is the best model so far based on validation accuracy
        if val_accuracy > best_val_accuracy:  # If the current validation accuracy is the best
            best_val_accuracy = val_accuracy  # Update best validation accuracy
            best_epoch = epoch  # Update best epoch
            torch.save(model.state_dict(), os.path.join(save_folder, f"best_{epoch+1}.pt"))  # Save the best model state
            if verbose: print(f"New best model saved with accuracy: {val_accuracy:.4f}\n")  # Print new best model message
        else:
            if verbose: print()  # Print newline

        torch.cuda.empty_cache()  # Empty CUDA cache

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))  # Create subplots for batch and epoch losses
    # Plot Batch Loss
    ax1.plot(batch_losses, label='Batch Loss')  # Plot batch losses
    ax1.set_xlabel('Batch number')  # Set x-axis label
    ax1.set_ylabel('Loss')  # Set y-axis label
    ax1.set_title('Training Loss Over Batches')  # Set plot title
    ax1.legend()  # Add legend
    # Plot Epoch Loss
    epoch_indices = np.arange(0, len(batch_losses), len(batch_losses) // num_epochs)  # Compute epoch indices
    ax2.plot(epoch_indices, epoch_losses, 'r', label='Epoch Loss')  # Plot epoch losses
    ax2.axvline(x=(best_epoch+1) * len(train_loader), color='g', linestyle='--', label='Best Model Epoch')  # Add vertical line for best epoch
    ax2.set_xlabel('Batch number')  # Set x-axis label
    ax2.set_ylabel('Loss')  # Set y-axis label
    ax2.set_title('Training Loss Over Epochs')  # Set plot title
    ax2.legend()  # Add legend
    # Save and show the figure
    plt.tight_layout()  # Adjust layout
    plt.savefig(os.path.join(save_folder, 'loss_graph.png'))  # Save the figure
    plt.close()  # Close the plot

    writer.close()  # Close the writer at the end

    torch.save(model.state_dict(), os.path.join(save_folder, 'final.pt'))  # Save the final model state
    return os.path.join(save_folder, f"best_{best_epoch+1}.pt")

def train_vit_model(images_folder_paths: list,
                    train_transform,
                    TRAIN_BATCH_SIZE: int, TEST_BATCH_SIZE: int,
                    training_folder_path: str,
                    use_partial_dataset: bool = False, dataset_percentage: float = 1.0,
                    freeze_backbone: bool = False, 
                    weighted_class: bool = False,
                    balanced: bool = True,
                    CUDA_DEVICE_IDS: list = None,
                    verbose=True):
    """
    Train a Vision Transformer (ViT) model.
    
    Parameters:
    - images_folder_paths: List of paths to image folders.
    - train_transform: Transformations to apply to training images.
    - TRAIN_BATCH_SIZE: Batch size for training.
    - TEST_BATCH_SIZE: Batch size for testing.
    - training_folder_path: Path to save training outputs.
    - use_partial_dataset: Flag to use a partial dataset.
    - dataset_percentage: Percentage of the dataset to use.
    - freeze_backbone: Flag to freeze the backbone of the model.
    - weighted_class: Flag to use class weights.
    - balanced: Flag to balance the dataset.
    - CUDA_DEVICE_IDS: List of CUDA device IDs.
    """
    model_specification_text = ("_balanced" if balanced else "") + ("_weighted" if weighted_class else "")  # Model specification text based on flags

    dataset, train_loader, val_loader = load_dataset(images_folder_paths, train_transform, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, balanced, use_partial_dataset, dataset_percentage)  # Load dataset

    dataset.class_distribution_graph.save(os.path.join(training_folder_path, "original_class_distribution.png"))  # Save class distribution graph
    dataset.image_size_distribution_graph.save(os.path.join(training_folder_path, "image_sizes_distribution.png"))  # Save image size distribution graph
    if balanced:  # If the dataset is balanced
        dataset.balanced_class_distribution_graph.save(os.path.join(training_folder_path, "balanced_class_distribution.png"))  # Save balanced class distribution graph

    num_classes = len(dataset.label_to_index)  # Number of classes
    model = vit_base_patch16_224(pretrained=True)  # Load pre-trained Vision Transformer model

    for param in model.parameters():  # Freeze or unfreeze model parameters
        param.requires_grad = not freeze_backbone

    model.head = nn.Sequential(
        nn.Linear(model.head.in_features, num_classes),  # Update the classifier head to match the number of classes
        nn.Softmax(dim=1)  # Add softmax activation
    )

    device = torch.device(f'cuda:{CUDA_DEVICE_IDS[0]}' if CUDA_DEVICE_IDS and torch.cuda.is_available() else 'cpu')  # Set device to GPU or CPU

    if len(CUDA_DEVICE_IDS) > 1 and torch.cuda.is_available():  # If multiple GPUs are available
        print(f"Using GPUs: {CUDA_DEVICE_IDS}")  # Print GPU IDs
        model = nn.DataParallel(model, device_ids=CUDA_DEVICE_IDS)  # Wrap model in DataParallel for multi-GPU training
    model.to(device)  # Move model to device

    if weighted_class:  # If using class weights
        print("Using weighted class")  # Print message
        class_weights = compute_class_weights(dataset.class_counts, dataset.label_to_index)  # Compute class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))  # Set loss function with class weights
    else:
        criterion = nn.CrossEntropyLoss()  # Set loss function without class weights

    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Initialize optimizer

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(training_folder_path, "runs"+model_specification_text))  # Initialize TensorBoard writer
    label_encoder = LabelEncoder()  # Initialize label encoder
    _ = label_encoder.fit_transform([label for _, label in dataset.samples])  # Fit label encoder to dataset labels
    class_names = label_encoder.classes_  # Get class names

    return train_model(model,  # Train the model
                train_loader, val_loader,
                criterion, optimizer, writer, class_names,
                device,
                num_epochs=2, start_from=0,
                save_folder=os.path.join(training_folder_path, "models"+model_specification_text),
                verbose=verbose)
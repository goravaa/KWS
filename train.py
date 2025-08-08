# train.py (Corrected and with TQDM)
#
# This script trains the lightweight KWSLiteCNN model for keyword spotting.
# FIX 1: Added the __main__ block to actually run the script.
# FIX 2: Integrated TQDM for a progress bar with ETA.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import os
import logging
import numpy as np
from tqdm import tqdm  # TQDM import

from model import KWSLiteCNN
from dataset import AudioDataset, get_mfcc_transform

def setup_logging():
    """Configures the logging for the training script for clean, informative output."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.StreamHandler()])

def run_epoch(model, dataloader, criterion, optimizer, device, is_training, epoch_num, total_epochs):
    """
    Runs a single epoch of training or validation with a TQDM progress bar.
    """
    model.train() if is_training else model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # TQDM progress bar setup
    desc = f"Epoch {epoch_num+1}/{total_epochs} - {'Training' if is_training else 'Validating'}"
    tqdm_bar = tqdm(dataloader, desc=desc, leave=False)

    with torch.set_grad_enabled(is_training):
        for inputs, labels in tqdm_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update TQDM progress bar with live stats
            tqdm_bar.set_postfix(loss=f"{(total_loss/total_samples):.4f}", acc=f"{(100*correct_predictions/total_samples):.2f}%")

    avg_loss = total_loss / total_samples
    accuracy = 100 * correct_predictions / total_samples
    return avg_loss, accuracy

def train_and_export():
    """Main function to orchestrate training, validation, and ONNX export."""
    setup_logging()
    
    # --- 1. Load Configuration and Setup ---
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model_dir = config['training']['model_output_path']
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, 'best_kws_model.pth')

    # --- 2. Prepare Data ---
    logging.info("Preparing data loaders...")
    mfcc_transform = get_mfcc_transform(config)
    
    # NOTE: Point this to your final dataset folder (e.g., 'dataset_augmented')
    dataset_path = config['training']['dataset_path'] 
    full_dataset = AudioDataset(config, dataset_path, transform=mfcc_transform)
    
    if len(full_dataset) == 0:
        logging.error(f"No audio files found in the dataset path: {dataset_path}. Please check your config.yaml and dataset folder.")
        return
        
    val_size = int(len(full_dataset) * config['training']['validation_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    loader_kwargs = {'num_workers': 2, 'pin_memory': True} if device.type == 'cuda' else {}
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, **loader_kwargs)
    logging.info(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples.")

    # --- 3. Initialize Model, Loss, Optimizer, and Scheduler ---
    model = KWSLiteCNN(num_classes=config['model']['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5)
    
    # --- 4. Training and Validation Loop ---
    best_val_accuracy = 0.0
    logging.info("Starting training...")
    num_epochs = config['training']['num_epochs']
    for epoch in range(num_epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True, epoch, num_epochs)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, False, epoch, num_epochs)
        
        scheduler.step(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch+1:02}/{num_epochs} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            logging.info(f"  -> New best validation accuracy: {val_acc:.2f}%. Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'classes': full_dataset.labels
            }, best_model_path)

    logging.info("Training finished.")

    # --- 5. Export to ONNX ---
    if not os.path.exists(best_model_path):
        logging.error("No best model was saved. Cannot export to ONNX. This might happen if the dataset is empty or training failed.")
        return
        
    logging.info("Exporting best model to ONNX format...")
    final_model = KWSLiteCNN(num_classes=config['model']['num_classes']).to(device)
    checkpoint = torch.load(best_model_path)
    final_model.load_state_dict(checkpoint['model_state_dict'])
    final_model.eval()

    cfg_audio = config['audio']
    time_steps = int((cfg_audio['sample_rate'] * cfg_audio['clip_duration_s']) / cfg_audio['hop_length']) + 1
    dummy_input = torch.randn(1, 1, cfg_audio['n_mfcc'], time_steps, device=device)
    
    onnx_path = os.path.join(model_dir, 'kws_model.onnx')
    torch.onnx.export(final_model, dummy_input, onnx_path, export_params=True,
                      opset_version=11, do_constant_folding=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    logging.info(f"Model successfully exported to {onnx_path}")
    logging.info(f"Associated classes from checkpoint: {checkpoint['classes']}")


# The crucial fix: Call the main function to start the script
if __name__ == "__main__":
    train_and_export()
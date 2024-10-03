import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from typing import Optional
from eval import validate
def train_one_epoch(model, train_loader, optimizer, device, scaler, use_mixed_precision, scheduler=None, scheduler_per_batch=False):
    """
    Train the model for one epoch, with an optional learning rate scheduler.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to train the model on (e.g., 'cuda' or 'cpu').
        scaler (GradScaler): Gradient scaler for mixed precision training.
        use_mixed_precision (bool): Whether to use mixed precision training.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Optional learning rate scheduler.
        scheduler_per_batch (bool): Whether to step the scheduler after each batch.

    Returns:
        tuple: (train_loss, train_accuracy)
    """
    model.train()
    total_train_loss, total_train, correct_train = 0, 0, 0

    # Initialize tqdm progress bar for training
    train_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch in train_bar:
        rich, poor, labels, _ = batch
        rich, poor, labels = rich.to(device), poor.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast(enabled=use_mixed_precision):
            outputs = model(rich, poor)
            loss = nn.CrossEntropyLoss()(outputs, labels)

        # Scale loss if using mixed precision
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_train_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        # Update tqdm bar with loss
        train_bar.set_postfix(loss=loss.item())
        
        # Step the scheduler if per batch
        if scheduler and scheduler_per_batch:
            scheduler.step()

    train_loss = total_train_loss / total_train
    train_accuracy = correct_train / total_train

    return train_loss, train_accuracy



def train_and_validate(model, 
                       train_loader, 
                       valid_loader, 
                       optimizer, 
                       device, 
                       num_epochs, 
                       best_model_path, 
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
                       scheduler_per_batch: bool = False,
                       use_mixed_precision: bool = True):
    """
    Train and validate the model with optional mixed precision and learning rate scheduler.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to train the model on (e.g., 'cuda' or 'cpu').
        num_epochs (int): Number of epochs to train.
        best_model_path (str): Path to save the best model.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Optional learning rate scheduler.
        scheduler_per_batch (bool): Whether to step the scheduler after each batch.
        use_mixed_precision (bool): Whether to use mixed precision training.

    Returns:
        None
    """
    # Set up mixed precision scaler if enabled
    scaler = GradScaler() if use_mixed_precision else None

    # Load checkpoint if available
    try:
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state'])
        best_val_accuracy = checkpoint['best_val_accuracy']
        print(f"Loaded previous best model with accuracy: {best_val_accuracy:.4f}")
    except FileNotFoundError:
        best_val_accuracy = float('-inf')
        print("No saved model found. Starting fresh!")

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

        # Train for one epoch
        train_loss, train_accuracy = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            device, 
            scaler, 
            use_mixed_precision, 
            scheduler=scheduler, 
            scheduler_per_batch=scheduler_per_batch
        )

        # Validate
        val_loss, val_accuracy = validate(model, valid_loader, device, use_mixed_precision)

        # Print overall training and validation accuracy
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        
        # Check if this is the best accuracy and save the model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({'model_state': model.state_dict(),
                        'best_val_accuracy': best_val_accuracy},
                       best_model_path)
            print(f"Saved new best model with accuracy: {best_val_accuracy:.4f}")

        # Step the scheduler if it's per epoch
        if scheduler and not scheduler_per_batch:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

    print("Training and validation complete.")

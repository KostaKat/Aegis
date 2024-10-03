#imports
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm


def validate(model, valid_loader, device, use_mixed_precision):
    """
    Validate the model.

    Args:
        model (nn.Module): The model to validate.
        valid_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to validate the model on (e.g., 'cuda' or 'cpu').
        use_mixed_precision (bool): Whether to use mixed precision validation.

    Returns:
        tuple: (val_loss, val_accuracy)
    """
    model.eval()
    total_val_loss, total_val, correct_val = 0, 0, 0

    # Initialize tqdm progress bar for validation
    valid_bar = tqdm(valid_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch in valid_bar:
            rich, poor, labels, _ = batch
            rich, poor, labels = rich.to(device), poor.to(device), labels.to(device)

            # Mixed precision inference
            with autocast(enabled=use_mixed_precision):
                outputs = model(rich, poor)
                loss = nn.CrossEntropyLoss()(outputs, labels)

            total_val_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            # Update tqdm bar with loss
            valid_bar.set_postfix(loss=loss.item())

    val_loss = total_val_loss / total_val if total_val > 0 else 1e-8
    val_accuracy = correct_val /  total_val if total_val > 0 else 1e-8

    return val_loss, val_accuracy

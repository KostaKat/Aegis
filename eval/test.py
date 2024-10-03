import torch
import numpy as np
from .eval_utils import display_confusion_matrices
def test(model,seen_models, test_loader, device, weights_path, name_model=None, include_unseen=True, save_path=None):
    """
    Test the model on the test_loader with optional separation of seen and unseen models.

    Args:
        model (nn.Module): The model to test.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').
        weights_path (str): Path to the model weights.
        name_model (str, optional): Name of the model (for logging purposes).
        include_unseen (bool): Whether to separate predictions for seen and unseen models.
    """
    try:
        # Load the best model
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval()
    seen_models = ["ADM", "BigGAN", "glide", "Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "VQDM", "wukong"]
    all_labels = []
    all_predictions = []
    per_model_labels = {}
    per_model_predictions = {}
    seen_labels = []
    seen_predictions = []
    unseen_labels = []
    unseen_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            rich, poor, labels, model_names = batch
            rich = rich.to(device)
            poor = poor.to(device)
            labels = labels.to(device)
            
            outputs = model(rich, poor)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            for model_name, label, prediction in zip(model_names, labels, predicted):
                # Separate seen and unseen models if include_unseen is True
                
                if model_name in seen_models:
                    seen_labels.append(label.item())
                    seen_predictions.append(prediction.item())
                elif include_unseen:
                    unseen_labels.append(label.item())
                    unseen_predictions.append(prediction.item())

                # Collect labels and predictions for per-model tracking
                if model_name not in per_model_labels:
                    per_model_labels[model_name] = []
                    per_model_predictions[model_name] = []
                per_model_labels[model_name].append(label.item())
                per_model_predictions[model_name].append(prediction.item())
        
    # Display confusion matrices (or other visualizations)
    if include_unseen:
        display_confusion_matrices(
            per_model_labels,
            per_model_predictions,
            np.array(all_labels),
            np.array(all_predictions),
            np.array(seen_labels),
            np.array(seen_predictions),
            np.array(unseen_labels),
            np.array(unseen_predictions),
            name_model,
            save_path=save_path
        )
    else:
        # If not including unseen, only display confusion matrices for all data
        display_confusion_matrices(
            per_model_labels,
            per_model_predictions,
            np.array(all_labels),
            np.array(all_predictions),
            np.array(seen_labels), np.array(seen_predictions), None, None,
            name_model,
            save_path=save_path
        )

import numpy as np  # For numerical operations
import pandas as pd  # For DataFrame creation and data manipulation
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For heatmap visualization
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score  # For evaluation metrics

def display_confusion_matrices(per_model_labels, per_model_predictions, all_labels, all_predictions, 
                               seen_labels, seen_predictions, unseen_labels, unseen_predictions, 
                               name_model, save_path=None):
    """
    Displays and saves confusion matrices and evaluation metrics for different model predictions.

    Args:
        per_model_labels (dict): Labels categorized by model.
        per_model_predictions (dict): Predictions categorized by model.
        all_labels (np.ndarray): All true labels across all data.
        all_predictions (np.ndarray): All predictions across all data.
        seen_labels (np.ndarray): True labels for 'seen' models.
        seen_predictions (np.ndarray): Predictions for 'seen' models.
        unseen_labels (np.ndarray): True labels for 'unseen' models.
        unseen_predictions (np.ndarray): Predictions for 'unseen' models.
        name_model (str): Name of the model being analyzed.
        save_path (str, optional): Path to save the confusion matrix image.

    Returns:
        None
    """
    metrics_data = []

    # Calculate metrics for each model
    for model_name, labels in per_model_labels.items():
        predictions = per_model_predictions[model_name]
        cm = confusion_matrix(labels, predictions)
        TN = cm[0, 0] if cm.shape[0] > 1 else 0
        FP = cm[0, 1] if cm.shape[0] > 1 else 0
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='macro', zero_division=0)
        
        metrics_data.append({
            "Model": model_name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "TN (Only AI Generated Image)": TN,
            "FP (Only AI Generated Image)": FP
        })

    # Create a DataFrame for model metrics
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create figure with subplots for confusion matrices and metrics
    fig, axs = plt.subplots(2, 3, figsize=(24, 12), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'{name_model} Confusion Matrices and Metrics', fontsize=16)

    # Plot overall, seen, and unseen confusion matrices
    plot_confusion_matrix(axs[0, 0], all_labels, all_predictions, 'Overall')
    plot_confusion_matrix(axs[0, 1], seen_labels, seen_predictions, 'Seen Models')
    if unseen_labels:
        plot_confusion_matrix(axs[0, 2], unseen_labels, unseen_predictions, 'Unseen Models')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Hide all axes for the lower row used for the table
    for ax in axs[1, :]:
        ax.axis('off')

    # Display metrics in a table
    axs[1, 1].axis('on')
    axs[1, 1].axis('tight')
    axs[1, 1].axis('off')
    table = axs[1, 1].table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')
    axs[1, 1].set_title('Metrics Per Model', pad=10, fontsize=14)
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.auto_set_column_width(list(range(len(metrics_df.columns))))
    
    # Style table headers
    for (i, key) in enumerate(metrics_df.columns):
        table[(0, i)].get_text().set_weight('bold')
        table[(0, i)].set_facecolor('#D3D3D3')

    # Final layout adjustments and saving the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    if save_path:
        plt.savefig(f'{save_path}{name_model}_confusion_matrix.png')

def get_label(i, j, labels, predictions):
    """
    Determines the label for confusion matrix cells.

    Args:
        i (int): Row index of the confusion matrix.
        j (int): Column index of the confusion matrix.
        labels (np.ndarray): Array of true labels.
        predictions (np.ndarray): Array of predictions.

    Returns:
        str: Label indicating True Positive (TP), True Negative (TN), False Positive (FP), or False Negative (FN).
    """
    if i == j:
        return 'TP' if i == 1 else 'TN'
    else:
        return 'FP' if i < j else 'FN'

def plot_confusion_matrix(ax, labels, predictions, title):
    """
    Plots a confusion matrix as a heatmap.

    Args:
        ax (matplotlib.axes.Axes): Axes object for the plot.
        labels (np.ndarray): Array of true labels.
        predictions (np.ndarray): Array of predictions.
        title (str): Title for the heatmap.

    Returns:
        None
    """
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=False, fmt='d', cmap='Purples', ax=ax, cbar=False)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    # Calculate threshold for text color
    threshold = cm.max() / 2.0

    # Add text annotations for each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.5, f"{get_label(i, j, labels, predictions)}\n({cm[i, j]})", 
                    ha="center", va="center", color="white" if cm[i, j] > threshold else "black", fontweight="bold")

    # Display accuracy and precision metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    ax.set_xlabel(f'Predicted\n\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}')

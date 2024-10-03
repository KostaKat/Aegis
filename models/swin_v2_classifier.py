import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinConfig, Swinv2ForImageClassification
from .cnn_block import CNNBlock


class SwinClassification(nn.Module):
    """
    A classification model that combines features from two inputs using CNN blocks
    and processes the feature difference through a Swin Transformer v2 for image classification.

    Args:
        kernels (torch.Tensor): High-pass filter kernels to be used in CNN blocks.

    Attributes:
        feature_combiner_rich (CNNBlock): CNN block for processing the 'rich' input.
        feature_combiner_poor (CNNBlock): CNN block for processing the 'poor' input.
        transformer (Swinv2ForImageClassification): Pretrained Swin Transformer model.
    """

    def __init__(self, kernels: torch.Tensor) -> None:
        super(SwinClassification, self).__init__()

        # Initialize CNN blocks for feature extraction
        self.feature_combiner_rich = CNNBlock(in_channels=30, out_channels=3, kernels=kernels)
        self.feature_combiner_poor = CNNBlock(in_channels=30, out_channels=3, kernels=kernels)

        # Load configuration for the Swin Transformer with specified number of classes
        config = SwinConfig.from_pretrained(
            'microsoft/swinv2-tiny-patch4-window8-256',
            num_labels=2  # Number of output classes
        )

        # Initialize the Swin Transformer for image classification
        self.transformer = Swinv2ForImageClassification.from_pretrained(
            'microsoft/swinv2-tiny-patch4-window8-256',
            ignore_mismatched_sizes=True,
            config=config
        )

        # Replace the classifier with a linear layer matching the desired output size
        self.transformer.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, rich: torch.Tensor, poor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SwinClassification model.

        Args:
            rich (torch.Tensor): Tensor representing the 'rich' input features of shape (batch_size, 30, H, W).
            poor (torch.Tensor): Tensor representing the 'poor' input features of shape (batch_size, 30, H, W).

        Returns:
            torch.Tensor: Logits output from the classifier of shape (batch_size, num_labels).
        """
        # Extract features from both inputs using CNN blocks
        rich_features = self.feature_combiner_rich(rich)
        poor_features = self.feature_combiner_poor(poor)

        # Compute the difference between the extracted features
        feature_difference = rich_features - poor_features

        # Pass the feature difference through the Swin Transformer
        outputs = self.transformer(feature_difference)

        return outputs.logits


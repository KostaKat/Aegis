#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from .high_pass_filter import HighPassFilters
class CNNBlock(nn.Module):
    """
    A convolutional block that applies high-pass filtering, followed by a convolution,
    batch normalization, and a Hardtanh activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernels (torch.Tensor): High-pass filter kernels for the HighPassFilters module.
    """

    def __init__(self, in_channels: int, out_channels: int, kernels: torch.Tensor) -> None:
        super(CNNBlock, self).__init__()
        self.filters = HighPassFilters(kernels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.Hardtanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNNBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying filtering, convolution, batch normalization, and activation.
        """
        x = self.filters(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

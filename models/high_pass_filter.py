# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
class HighPassFilters(nn.Module):
    """
    Applies high-pass filtering to input images using predefined kernels.

    Args:
        kernels (torch.Tensor): A tensor of shape (out_channels, in_channels, kernel_height, kernel_width)
                                 representing the high-pass filter kernels. These kernels are not trainable.
    """

    def __init__(self, kernels: torch.Tensor) -> None:
        super(HighPassFilters, self).__init__()
        if kernels.requires_grad:
            raise ValueError("Kernels should not require gradients.")
        self.register_buffer('kernels', kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying high-pass filters to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Filtered tensor after applying high-pass convolution.
        """
        padding = (self.kernels.shape[2] // 2, self.kernels.shape[3] // 2)
        return F.conv2d(x, self.kernels, padding=padding)



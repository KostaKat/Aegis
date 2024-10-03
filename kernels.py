import numpy as np
import torch
from scipy.ndimage import rotate

kernels = np.array([np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]), np.array([
    [0, 0, -1, 0, 0],
    [0, 0, 3, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0]
]), np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, -2, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0]
]), np.array([
    [0, 0, 0, 0, 0],
    [0, -1, 2, -1, 0],
    [0, 2, -4, 2, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]), np.array([
    [-1, 2, -2, 2, -1],
    [2, -6, 8, -6, 2],
    [-2, 8, -12, 8, -2],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]), np.array([
    [0, 0, 0, 0, 0],
    [0, -1, 2, -1, 0],
    [0, 2, -4, 2, 0],
    [0, -1, 2, -1, 0],
    [0, 0, 0, 0, 0]
]), np.array([
    [-1, 2, -2, 2, -1],
    [2, -6, 8, -6, 2],
    [-2, 8, -12, 8, -2],
    [2, -6, 8, -6, 2],
    [-1, 2, -2, 2, -1],
])])

angles = [
    [45, 90, 135, 180, 225, 270, 315, 0],
    [0, 45, 90, 135, 180, 225, 270, 315],
    [90, 180, 45, 135],
    [90, 180, 270, 0],
    [90, 180, 270, 0],
    [0],
    [0]
]
def apply_high_pass_filter():
    rotated_kernels = []

    for idx, kernel in enumerate(kernels):
        for angle in angles[idx]:
            # Rotate kernel
            rotated_kernel = rotate(kernel, angle, reshape=False)
            # Ensure the kernel is in float32 format
            rotated_kernel = np.round(rotated_kernel).astype(np.float32)
            # Convert to tensor, shape [5, 5]
            tensor_kernel = torch.tensor(rotated_kernel)
            # Unsqueeze and repeat to convert to 3-channel, shape [3, 5, 5]
            tensor_kernel = tensor_kernel.unsqueeze(0).repeat(3, 1, 1)
            rotated_kernels.append(tensor_kernel)

    # Stack all kernels to form a single tensor [num_kernels * num_angles, 3, 5, 5]
    all_kernels = torch.stack(rotated_kernels)

    return all_kernels

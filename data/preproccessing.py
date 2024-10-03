import numpy as np  # For numerical computations
import random  # For random operations
from PIL import Image  # For image processing
import cv2  # For OpenCV functionalities (e.g., color conversion)
import torch  # For tensor manipulations
from scipy.ndimage import rotate  # For image rotation

def img_to_patches(img, min_patches=128, patch_size=32) -> tuple:
    """
    Splits the given image into smaller patches of a specified size.

    Args:
        img (PIL.Image): The input image to be split into patches.
        min_patches (int): Minimum number of patches required.
        patch_size (int): The size of each square patch.

    Returns:
        tuple: Grayscale and color patches extracted from the image.
    """
    target_patches = (256 // patch_size) ** 2
    current_patches_x = img.size[0] // patch_size
    current_patches_y = img.size[1] // patch_size
    current_total_patches = current_patches_x * current_patches_y

    # Resize the image if it has fewer patches than a 256x256 image
    if current_total_patches < target_patches:
        img = img.resize((max(256, img.size[0]), max(256, img.size[1])))

    grayscale_patches = []
    color_patches = []

    for i in range(0, img.height, patch_size):
        for j in range(0, img.width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = np.asarray(img.crop(box))
            grayscale_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            grayscale_patches.append(grayscale_patch.astype(np.int32))
            color_patches.append(patch)

    return grayscale_patches, color_patches

def get_l1(v):
    """Calculates the L1 variation of the given patch."""
    return np.sum(np.abs(v[:, :-1] - v[:, 1:]))

def get_l2(v):
    """Calculates the L2 variation of the given patch."""
    return np.sum(np.abs(v[:-1, :] - v[1:, :]))

def get_l3l4(v):
    """Calculates diagonal variations of the given patch."""
    l3 = np.sum(np.abs(v[:-1, :-1] - v[1:, 1:]))
    l4 = np.sum(np.abs(v[1:, :-1] - v[:-1, 1:]))
    return l3 + l4

def get_pixel_var_degree_for_patch(patch):
    """
    Calculates the pixel variance degree for a given patch.

    Args:
        patch (np.ndarray): The image patch.

    Returns:
        float: The calculated variance degree.
    """
    l1 = get_l1(patch)
    l2 = get_l2(patch)
    l3l4 = get_l3l4(patch)
    return l1 + l2 + l3l4

def duplicate_to_minimum_sorted(patches, variances, min_count=64):
    """
    Ensures at least a minimum number of patches by duplicating existing ones.

    Args:
        patches (list): List of patches.
        variances (list): List of variances corresponding to the patches.
        min_count (int): Minimum number of patches required.

    Returns:
        tuple: Duplicated patches and their variances.
    """
    if len(patches) < min_count:
        paired = sorted(zip(patches, variances), key=lambda x: x[1], reverse=True)
        while len(paired) < min_count:
            random.shuffle(paired)
            additional_needed = min_count - len(paired)
            paired.extend(paired[:additional_needed])
        patches, variances = zip(*paired)
    return list(patches), list(variances)

def extract_rich_and_poor_textures(variance_values, patches):
    """
    Splits the patches into rich and poor textures based on their variance.

    Args:
        variance_values (list): List of variances for the patches.
        patches (list): List of patches.

    Returns:
        tuple: Rich and poor texture patches.
    """
    sorted_indices = np.argsort(variance_values)[::-1]
    sorted_patches = [patches[i] for i in sorted_indices]
    sorted_variances = [variance_values[i] for i in sorted_indices]

    if len(patches) < 192:
        threshold = np.mean(variance_values)
        rich_patches = [patch for patch, var in zip(sorted_patches, sorted_variances) if var >= threshold]
        rich_variances = [var for var in sorted_variances if var >= threshold]
        poor_patches = [patch for patch, var in zip(sorted_patches, sorted_variances) if var < threshold]
        poor_variances = [var for var in sorted_variances if var < threshold]

        # Ensure each category has at least 64 patches while maintaining sorted order by variance
        rich_patches, rich_variances = duplicate_to_minimum_sorted(rich_patches, rich_variances, 64)
        poor_patches, poor_variances = duplicate_to_minimum_sorted(poor_patches, poor_variances, 64)
    else:
        num_top_patches = len(patches) // 3
        rich_patches = [patches[i] for i in sorted_indices[:num_top_patches]]
        poor_patches = [patches[i] for i in sorted_indices[-num_top_patches:]]

    return rich_patches, poor_patches

def get_complete_image(patches, coloured=True):
    """
    Reconstructs a complete image from its patches.

    Args:
        patches (list): List of patches.
        coloured (bool): Whether the patches are colored.

    Returns:
        np.ndarray: Reconstructed image from patches.
    """
    patches = patches[:64]
    grid = np.array(patches).reshape((8, 8, 32, 32, 3)) if coloured else np.array(patches).reshape((8, 8, 32, 32))
    rows = [np.concatenate(row_patches, axis=1) for row_patches in grid]
    complete_image = np.concatenate(rows, axis=0)
    return complete_image

def smash_n_reconstruct(input_path, coloured=True):
    """
    Splits an image into patches, extracts textures, and reconstructs images based on textures.

    Args:
        input_path (str): Path to the input image.
        coloured (bool): Whether to process the image as colored.

    Returns:
        tuple: Reconstructed rich and poor texture images.
    """
    grayscale_patches, color_patches = img_to_patches(input_path)
    pixel_var_degree = [get_pixel_var_degree_for_patch(patch) for patch in grayscale_patches]

    if coloured:
        rich_patches, poor_patches = extract_rich_and_poor_textures(pixel_var_degree, color_patches)
    else:
        rich_patches, poor_patches = extract_rich_and_poor_textures(pixel_var_degree, grayscale_patches)

    rich_texture = get_complete_image(rich_patches, coloured)
    poor_texture = get_complete_image(poor_patches, coloured)

    return rich_texture, poor_texture

import random  # For generating random numbers
import io  # For in-memory file operations
from PIL import Image, ImageFilter  # For image processing and filtering
from typing import Tuple  # For type hinting tuples
import torchvision.transforms as T
class JpegCompression:
    """
    Applies JPEG compression to an image with a certain probability. The quality of compression is randomly chosen
    within a specified range to simulate compression artifacts.
    """
    def __init__(self, prob: float = 0.1, quality_range: Tuple[int, int] = (70, 100)):
        """
        Initializes the JpegCompression object.

        Args:
            prob (float): The probability of applying JPEG compression to an image.
            quality_range (Tuple[int, int]): The range of JPEG quality to apply if compression is used.
        """
        self.prob = prob  # Probability of applying the transformation
        self.quality_range = quality_range  # Range for JPEG compression quality

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Applies JPEG compression to the image based on the specified probability.

        Args:
            img (Image.Image): The input image to be transformed.

        Returns:
            Image.Image: The transformed image with JPEG compression applied.
        """
        if random.random() < self.prob:  # Only apply compression if the random number is below probability
            quality = random.randint(*self.quality_range)  # Randomly choose quality within the range
            img_bytes = io.BytesIO()  # Create an in-memory byte buffer
            img.save(img_bytes, format='JPEG', quality=quality)  # Save the image with JPEG compression
            img_bytes.seek(0)  # Rewind the buffer to the beginning
            img = Image.open(img_bytes).convert("RGB")  # Load the image from the buffer as RGB
        return img  # Return the transformed or original image

    def __repr__(self):
        """Provides a string representation of the transformation settings."""
        return f"{self.__class__.__name__}(prob={self.prob}, quality_range={self.quality_range})"


class GaussianBlur:
    """
    Applies Gaussian blur to an image with a certain probability. The blur is applied with a random sigma
    chosen from the specified range.
    """
    def __init__(self, prob: float = 0.1, sigma_range: Tuple[float, float] = (0, 1)):
        """
        Initializes the GaussianBlur object.

        Args:
            prob (float): The probability of applying Gaussian blur to an image.
            sigma_range (Tuple[float, float]): The range of standard deviation (sigma) for the blur.
        """
        self.prob = prob  # Probability of applying the transformation
        self.sigma_range = sigma_range  # Range for the sigma (standard deviation) of the Gaussian blur

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Applies Gaussian blur to the image based on the specified probability.

        Args:
            img (Image.Image): The input image to be transformed.

        Returns:
            Image.Image: The transformed image with Gaussian blur applied.
        """
        if random.random() < self.prob:  # Only apply blur if the random number is below probability
            sigma = random.uniform(*self.sigma_range)  # Randomly choose sigma within the range
            img = img.filter(ImageFilter.GaussianBlur(sigma))  # Apply Gaussian blur to the image
        return img  # Return the transformed or original image

    def __repr__(self):
        """Provides a string representation of the transformation settings."""
        return f"{self.__class__.__name__}(prob={self.prob}, sigma_range={self.sigma_range})"


class Downsampling:
    """
    Applies downsampling and then upsampling to an image with a certain probability to simulate image degradation.
    The downsampling scale is randomly chosen from the specified range.
    """
    def __init__(self, prob: float = 0.1, scale_range: Tuple[float, float] = (0.25, 0.5)):
        """
        Initializes the Downsampling object.

        Args:
            prob (float): The probability of applying downsampling to an image.
            scale_range (Tuple[float, float]): The range of scales to downsample the image before upsampling back.
        """
        self.prob = prob  # Probability of applying the transformation
        self.scale_range = scale_range  # Range for the downsampling scale

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Applies downsampling and then upsampling to the image based on the specified probability.

        Args:
            img (Image.Image): The input image to be transformed.

        Returns:
            Image.Image: The transformed image with downsampling and upsampling applied.
        """
        if random.random() < self.prob:  # Only apply downsampling if the random number is below probability
            scale = random.uniform(*self.scale_range)  # Randomly choose downsampling scale within the range
            # Downsample the image by scaling it down
            smaller_img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BICUBIC)
            # Upsample the image back to its original size
            img = smaller_img.resize((img.width, img.height), Image.BICUBIC)
        return img  # Return the transformed or original image

    def __repr__(self):
        """Provides a string representation of the transformation settings."""
        return f"{self.__class__.__name__}(prob={self.prob}, scale_range={self.scale_range})"
class RandomResize:
    def __init__(self, resize_range, prob=0.5):
        self.resize_range = resize_range
        self.prob = prob  # Probability of applying the resize

    def __call__(self, img):
        # Randomly decide whether to apply resize
        if random.random() < self.prob:
            random_size = random.randint(self.resize_range[0], self.resize_range[1])
            return T.Resize((random_size, random_size))(img)
        return img  # Return the image without resizing
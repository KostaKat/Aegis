import os  # For directory operations
import numpy as np  # For numerical operations
from torch.utils.data import Dataset, Subset  # PyTorch classes for handling datasets
from collections import defaultdict  # For dictionary operations with default values
from PIL import Image  # For image processing
from .preproccessing import smash_n_reconstruct
class DatasetAI(Dataset):
    """
    Custom PyTorch Dataset for AI-generated and nature images.

    Args:
        root_dir (str): The root directory containing image data.
        transforms_pre (callable, optional): A function/transform to apply to the images before preprocessing.
        transforms_post (callable, optional): A function/transform to apply to the images after preprocessing.
        split (str): Dataset split ('train', 'val', 'test').
    """
    def __init__(self, root_dir, transforms_pre=None, transforms_post=None, split='train'):
        self.root_dir = root_dir
        self.transforms_pre = transforms_pre
        self.transforms_post = transforms_post
        self.split = split
        self.samples = []  # List to store image paths and associated metadata
        self.file_paths = []  # To track all file paths for duplicate checks

        # Load samples from each model directory
        for model_name in sorted(os.listdir(root_dir)):
            model_path = os.path.join(root_dir, model_name)
            if os.path.isdir(model_path):
                imagenet_dir = f'imagenet_{model_name}'
                data_dir = os.path.join(model_path, imagenet_dir, split)
                
                # Collect class-specific samples (AI or nature images)
                if os.path.isdir(data_dir):
                    for class_label in ['ai', 'nature']:
                        class_path = os.path.join(data_dir, class_label)
                        if os.path.exists(class_path):
                            for img_name in os.listdir(class_path):
                                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    img_path = os.path.join(class_path, img_name)
                                    self.samples.append((img_path, class_label, model_name))
                                    self.file_paths.append(img_path)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Rich and poor texture images, label (0 for 'ai', 1 for 'nature'), and model name.
        """
        img_path, class_label, model_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transformations if any
        if self.transforms_pre:
            image = self.transforms_pre(image)

        # Preprocess the image to generate rich and poor textures
        rich, poor = smash_n_reconstruct(image)
        if self.transforms_post:
            rich = self.transforms_post(rich)
            poor = self.transforms_post(poor)
        # Assign label based on class
        label = 0 if class_label == 'ai' else 1

        return rich, poor, label, model_name

def split_datasets(train_dataset, val_test_dataset, train_size, val_size, test_size, seed_train=42, seed_val=42, seed_test=42):
    """
    Splits the datasets into training, validation, and test subsets while balancing class distribution.

    Args:
        train_dataset (Dataset): The dataset for training.
        val_test_dataset (Dataset): The dataset for validation and testing.
        train_size (int): Number of training samples.
        val_size (int): Number of validation samples.
        test_size (int): Number of test samples.
        seed_train (int): Random seed for training set split.
        seed_val (int): Random seed for validation set split.
        seed_test (int): Random seed for test set split.

    Returns:
        tuple: Subsets for training, validation, and testing.
    """
    rng_train = np.random.default_rng(seed_train)
    rng_val = np.random.default_rng(seed_val)
    rng_test = np.random.default_rng(seed_test)

    # Dictionary to store indices for training and validation per model
    indices_dict = defaultdict(lambda: {'train': [], 'val': []})
   
    # Collect indices for training and validation datasets
    for idx, (_, class_label, model_name) in enumerate(train_dataset.samples):
        indices_dict[model_name]['train'].append(idx)
    
    for idx, (_, class_label, model_name) in enumerate(val_test_dataset.samples):
        indices_dict[model_name]['val'].append(idx)

    # Calculate the size of splits per model to ensure balanced classes
    num_models = len(indices_dict)
    model_train_size = train_size // num_models
    model_val_size = val_size // num_models
    model_test_size = test_size // num_models

    aggregated_train_indices = []
    aggregated_val_indices = []
    aggregated_test_indices = []

    # Balance and shuffle indices for each model
    for model_name, indices in indices_dict.items():
        train_indices = np.array(indices['train'])
        val_indices = np.array(indices['val'])

        rng_train.shuffle(train_indices)
        rng_val.shuffle(val_indices)

        # Balance training indices
        if len(train_indices) >= model_train_size:
            train_balanced_indices = rng_train.choice(train_indices, size=model_train_size, replace=False)
            aggregated_train_indices.extend(train_balanced_indices)

        # Allocate indices for validation and test sets
        if len(val_indices) >= model_val_size + model_test_size:
            val_balanced_indices = val_indices[:model_val_size]
            test_balanced_indices = val_indices[model_val_size:model_val_size + model_test_size]
        else:
            split_index = int(len(val_indices) * (model_val_size / (model_val_size + model_test_size)))
            val_balanced_indices = val_indices[:split_index]
            test_balanced_indices = val_indices[split_index:split_index + model_test_size]

        aggregated_val_indices.extend(val_balanced_indices)
        rng_test.shuffle(test_balanced_indices)
        aggregated_test_indices.extend(test_balanced_indices)

    # Create and return subsets for train, validation, and test
    train_subset = Subset(train_dataset, aggregated_train_indices)
    val_subset = Subset(val_test_dataset, aggregated_val_indices)
    test_subset = Subset(val_test_dataset, aggregated_test_indices)
    
    return train_subset, val_subset, test_subset

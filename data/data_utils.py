
from torch.utils.data import Dataset  # If you're working with PyTorch Datasets
from collections import Counter  # For counting occurrences

def print_model_class_distribution(dataset, indices=None):
    """
    Prints the distribution of model and class combinations in a dataset or a subset.
    
    Args:
        dataset (Dataset): The dataset or subset to analyze.
        indices (list of int, optional): Indices to analyze within the dataset. If None, analyze the entire dataset.
    """
    model_class_counter = Counter()
    total_samples = 0

    # Determine which samples to count
    if indices is None:
        samples_to_count = dataset.samples
    else:
        samples_to_count = [dataset.samples[i] for i in indices]

    # Count each combination of model and class
    for _, class_label, model_name in samples_to_count:
        model_class_counter[(model_name, class_label)] += 1
        total_samples += 1

    # Print the counted distributions
    print(f"Total samples in subset: {total_samples}")
    for (model, class_label), count in model_class_counter.items():
        percentage = (count / total_samples) * 100
        print(f"Model {model}, Class {class_label}: {count} ({percentage:.2f}%)")
def check_data_overlap(subset1, subset2):
    paths1 = {subset1.dataset.file_paths[i] for i in subset1.indices}
    paths2 = {subset2.dataset.file_paths[i] for i in subset2.indices}
    overlap = paths1.intersection(paths2)
    if overlap:
        print(f"Actual data overlap detected with {len(overlap)} items.")
        return overlap
    else:
        print("No actual data overlap detected.")
        return set()
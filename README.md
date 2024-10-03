# Aegis Release

This repository contains the code and resources for the Aegis project.

## Contents
- `Aegis_AI_Detection.pdf`: A comprehensive document detailing the implementation of the model, including design choices, training process, and evaluation results.

## How to Use
- Our weights can be found here: [Link to google drive](https://drive.google.com/file/d/1_Ezqe_Y9D6o03z8PFYDtD34oJD3gz-Cj/view?usp=sharing)
- To understand the implementation details, refer to `Aegis_AI_Detection.pdf`.

## Repository Structure
- `data/`: Directory containing the data utilities
- `models/`: Model architectures and utility functions.
- `train/`: Code for training the model.
- `eval/`: Code for evaluating the trained model.

- `requirements.txt`: List of dependencies required for running the code.

## Getting Started
1. Install the dependencies:
   ```bash
   pip install -r requirements.txt

2. Run main.py with the desired arguments
```bash
python main.py [arguments]
```
Arguments

Data Arguments

--root_dir: Path to the root directory of datasets (required) \
--batch_size: Batch size for DataLoader (default: 32). \
--num_workers: Number of workers for DataLoader (default: 4).\
--train_size: Size of training data (default: 1000).\
--val_size: Size of validation data (default: 200).\
--test_size: Size of test data (default: 200). 

Training Arguments

--step_size: Step size for learning rate scheduler (default: 10). \
--gamma: Gamma value for learning rate scheduler (default: 0.1).\
--learning_rate: Learning rate for training (default: 1e-3).\
--num_epochs: Number of epochs to train (default: 10).\
--seed_train: Seed for training data splitting (default: 42).\
--seed_val: Seed for validation data splitting (default: 42).\
--seed_test: Seed for test data splitting (default: 42).\
--pre_aug: Apply pre-transformations before training.\
--device: Device for model operations (e.g., cpu or cuda) (default: cuda).\
--use_mixed_precision: Enable mixed precision training.\
--weights_path: Path to load or save model weights (default: None).\
--train: Flag to enable training mode.\
--test: Flag to enable testing mode.\
--include_unseen: Include unseen models in test results.\
--name_model: Name of the model for logging purposes (default: "model").\
--scheduler_per_batch: Apply scheduler per batch instead of per epoch.\
--save_path: Path to save evaluation results (default: results/).

Example command
```bash 
python main.py --root_dir /path/to/data --train --batch_size 64 --learning_rate 0.001 --num_epochs 20
```

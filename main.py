import argparse
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from data.dataset import DatasetAI, split_datasets
from data.transforms import JpegCompression, GaussianBlur, Downsampling, RandomResize
from train.train import train_and_validate
from models.swin_v2_classifier import SwinClassification
from kernels import apply_high_pass_filter
from eval.test import test

def main():
    """Main function to execute the data processing, training, and testing pipeline."""

    # Parse command-line arguments
    args = get_arguments()

    # Set the device for computation
    device = torch.device(args.device)

    # Define data transformations
    transforms_pre = None
    resize_range = (150, 2024)  
    if args.pre_aug:
        transforms_pre = T.Compose([
            JpegCompression(),
            GaussianBlur(),
            Downsampling(),
            RandomResize(resize_range, prob=0.3),  # 50% chance to resize randomly within the range
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.RandomHorizontalFlip(),
        ])

    transforms_post = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets for train and validation/test
    train_dataset = DatasetAI(
        root_dir=args.root_dir, 
        transforms_post=transforms_post, 
        transforms_pre=transforms_pre, 
        split='train'
    )
    val_test_dataset = DatasetAI(
        root_dir=args.root_dir, 
        transforms_post=transforms_post, 
        transforms_pre=None, 
        split='val'
    )

    # Split datasets into train, validation, and test subsets
    if args.test:
        train_subset, val_subset, test_subset = split_datasets(
            train_dataset, 
            val_test_dataset, 
            args.train_size, 
            args.val_size, 
            args.test_size
        )
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize model
    model = SwinClassification(kernels=apply_high_pass_filter()).to(device)
    
    # Train the model if specified
    if args.train:
        print("Training mode enabled.")
        
        # Unfreeze model parameters for training
        for param in model.parameters():
            param.requires_grad = True
        
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        train_and_validate(
            model=model,
            train_loader=train_loader,
            valid_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=args.num_epochs,
            best_model_path=args.weights_path,
            scheduler=scheduler,
            scheduler_per_batch=args.scheduler_per_batch,
            use_mixed_precision=args.use_mixed_precision
        )
        print("Training complete.")
    
    # Test the model if specified
    if args.test:
        print("Testing mode enabled.")
        print("Including unseen models in test results.")
        test(
            model=model,
            seen_models=["ADM", "BigGAN", "glide", "Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "VQDM", "wukong"],
            test_loader=test_loader,
            device=device,
            weights_path=args.weights_path,
            name_model=args.name_model,
            include_unseen=args.include_unseen,
            save_path=args.save_path
        )

def get_arguments():
    """Parse and return command-line arguments."""
    
    parser = argparse.ArgumentParser(description="Run data processing, augmentation, and model training/testing pipeline.")

    # Data arguments
    parser.add_argument('--root_dir', type=str, required=True, help='Path to the root directory of datasets.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader.')
    parser.add_argument('--train_size', type=int, default=1000, help='Size of training data.')
    parser.add_argument('--val_size', type=int, default=200, help='Size of validation data.')
    parser.add_argument('--test_size', type=int, default=200, help='Size of test data.')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma value for learning rate scheduler.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for training.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--seed_train', type=int, default=42, help='Seed for training data splitting.')
    parser.add_argument('--seed_val', type=int, default=42, help='Seed for validation data splitting.')
    parser.add_argument('--seed_test', type=int, default=42, help='Seed for test data splitting.')
    parser.add_argument('--pre_aug', action='store_true', help='Apply pre-transformations before training.')
    parser.add_argument('--device', type=str, default='cuda', help='Device for model operations (e.g., cpu or cuda).')
    parser.add_argument('--use_mixed_precision', action='store_true', help='Enable mixed precision training.')
    parser.add_argument('--weights_path', type=str, default=None, help='Path to load or save model weights.')
    parser.add_argument('--train', action='store_true', help='Flag to enable training mode.')
    parser.add_argument('--test', action='store_true', help='Flag to enable testing mode.')
    parser.add_argument('--include_unseen', action='store_true', help='Include unseen models in test results.')
    parser.add_argument('--name_model', type=str, default="model", help='Name of the model for logging purposes.')
    parser.add_argument('--scheduler_per_batch', action='store_true', help='Apply scheduler per batch instead of per epoch.')
    parser.add_argument('--save_path', type=str, default='results/', help='Path to save evaluation results.')

    return parser.parse_args()

if __name__ == "__main__":
    main()

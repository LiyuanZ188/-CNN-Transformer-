import argparse
import yaml
import torch
from datasets import get_dataloader
from models import get_model
from training import train_model
from evaluation import evaluate_model

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate models with optional data augmentation.")
    parser.add_argument("-dataset", type=str, required=True, help="Dataset to use (e.g., 'cifar10').")
    parser.add_argument("-model", type=str, required=True, help="Model to use (e.g., 'resnet').")
    parser.add_argument("-augmentation", type=str, default=None,
                        help="Comma-separated list of data augmentation techniques (e.g., 'flip,rotate,augmix').")
    parser.add_argument("-epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size for training and testing.")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("-weight_decay", type=float, default=0.0, help="Weight decay for regularization.")
    parser.add_argument("-momentum", type=float, default=0.9, help="Momentum for the optimizer.")
    parser.add_argument("-optimizer", type=str, default="adam", help="Optimizer to use (e.g., 'adam' or 'sgd').")
    parser.add_argument("-patience", type=int, default=5, help="Number of epochs to wait for improvement before early stopping.")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the trained model weights.")
    args = parser.parse_args()

    # Map dataset names to number of classes
    dataset_classes = {
        "cifar10": 10,
        "cifar100": 100,
        "mnist": 10,
        "fashion_mnist": 10
    }

    if args.dataset not in dataset_classes:
        raise ValueError(f"Dataset {args.dataset} is not supported.")
    num_classes = dataset_classes[args.dataset]
    print(f"Using dataset {args.dataset} with {num_classes} classes.")

    # Do not pass sample-level augmentation to the dataloader.
    # The dataloader should only perform basic preprocessing.
    train_transform = None
    train_loader, val_loader, test_loader = get_dataloader(
        dataset_name=args.dataset,
        model_name=args.model,
        train_transform=train_transform,
        split_ratio=0.8,
        batch_size=args.batch_size
    )

    if args.augmentation and args.augmentation.lower() != "none":
        print(f"Data augmentation enabled: {args.augmentation}")
    else:
        print("No data augmentation applied.")

    # Get and set up the model
    model = get_model(args.model, num_classes=num_classes)
    model.to(device)

    # Train the model. The augmentations will be applied inside train_model..
    print("Starting training...")
    best_model = train_model(
        model,
        train_loader,
        num_classes=num_classes,
        augmentation_name=args.augmentation,
        validation_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        optimizer_type=args.optimizer,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        patience=args.patience,
        save_path=args.save_path
    )

    # Evaluate the model using the test dataset
    print("Starting evaluation...")
    best_model.eval()
    accuracy, precision, recall, f1 = evaluate_model(best_model, test_loader, num_classes)

    # Convert tensor results to Python scalars before printing
    accuracy = accuracy.cpu().item() if torch.is_tensor(accuracy) else accuracy
    precision = precision.cpu().item() if torch.is_tensor(precision) else precision
    recall = recall.cpu().item() if torch.is_tensor(recall) else recall
    f1 = f1.cpu().item() if torch.is_tensor(f1) else f1

    print(f"Final Test Metrics:\n"
          f"Accuracy: {accuracy:.4f}\n"
          f"Precision: {precision:.4f}\n"
          f"Recall: {recall:.4f}\n"
          f"F1 Score: {f1:.4f}\n")

if __name__ == "__main__":
    main()



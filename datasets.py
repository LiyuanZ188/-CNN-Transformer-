from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, datasets
import torch


def get_dataloader(dataset_name, model_name, train_transform=None, split_ratio=0.8, batch_size=32):

    dataset_map = {
        "cifar10": (datasets.CIFAR10, True),
        "cifar100": (datasets.CIFAR100, True),
        "mnist": (datasets.MNIST, True),
        "fashion_mnist": (datasets.FashionMNIST, True)
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    dataset_class, has_official_split = dataset_map[dataset_name]

    # Use ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Base transformations: resize and convert to tensor (with grayscale conversion for 1-channel datasets)
    base_transforms = [transforms.Resize((224, 224))]
    if dataset_name in ["mnist", "fashion_mnist"]:
        # Convert grayscale images to 3-channel before normalization
        base_transforms.append(transforms.Grayscale(num_output_channels=3))
    base_transforms.append(transforms.ToTensor())
    # Define the basic test transform (no augmentation, just preprocessing and normalization)
    test_transform = transforms.Compose(base_transforms + [normalize])

    if has_official_split:
        # For datasets with an official train/test split
        # Use training set for train/val split and provided test set as test_dataset
        full_train_dataset_noaug = dataset_class(root="data", train=True, download=True, transform=test_transform)
        if train_transform:
            train_aug_pipeline = transforms.Compose(base_transforms + [train_transform, normalize])
        else:
            train_aug_pipeline = test_transform
        full_train_dataset_aug = dataset_class(root="data", train=True, download=True, transform=train_aug_pipeline)
        test_dataset = dataset_class(root="data", train=False, download=True, transform=test_transform)
        # Split training set into train and validation subsets (fixed seed for reproducibility)
        num_train = len(full_train_dataset_noaug)
        train_size = int(split_ratio * num_train)
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(num_train, generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_dataset = Subset(full_train_dataset_aug, train_indices)
        val_dataset = Subset(full_train_dataset_noaug, val_indices)
    else:
        # For datasets without an official split, perform a random split into train/val/test
        full_dataset = dataset_class(root="data", transform=test_transform)
        total_len = len(full_dataset)
        train_size = int(split_ratio * total_len)
        val_size = int((1 - split_ratio) / 2 * total_len)
        test_size = total_len - train_size - val_size
        # Use a fixed seed for reproducible splits
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



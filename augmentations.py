import torchvision.transforms as transforms
import torch
import numpy as np

def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size)

    # Randomly choose a center for the patch
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    # Compute patch size
    r = np.sqrt(1 - lam)
    cut_w = int(w * r)
    cut_h = int(h * r)

    # Get bounding box coordinates
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    # Replace the patch in the current image with the patch from another random image in the batch
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # Recompute lambda based on the patch area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def get_augmentation(augmentation_name):

    if augmentation_name is None or augmentation_name.lower() == "none":
        return {"use_mixup": False, "use_cutmix": False, "sample_transform": None}

    # Mapping for sample-level transformations
    aug_map = {
        "flip": transforms.RandomHorizontalFlip(),
        "rotate": transforms.RandomRotation(10),
        "color_jitter": transforms.ColorJitter(brightness=0.2, contrast=0.2),
        "gaussian_blur": transforms.GaussianBlur(kernel_size=(3, 3)),
        "random_erasing": transforms.RandomErasing(),
        "cutout": transforms.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        "random_resized_crop": transforms.RandomResizedCrop(size=224),
        "affine": transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        "random_perspective": transforms.RandomPerspective(distortion_scale=0.5, p=0.5)
    }

    use_mixup = False
    use_cutmix = False
    sample_transforms = []

    for aug in augmentation_name.split(","):
        aug = aug.strip().lower()
        if aug == "mixup":
            use_mixup = True
        elif aug == "cutmix":
            use_cutmix = True
        elif aug == "autoaugment":
            # Wrap AutoAugment in a Compose that converts to PIL and back.
            sample_transforms.append(transforms.Compose([
                transforms.ToPILImage(),
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor()
            ]))
        elif aug == "randaugment":
            # wrap RandAugment.
            sample_transforms.append(transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandAugment(),
                transforms.ToTensor()
            ]))
        elif aug == "augmix":
            # AugMix augmentation
            sample_transforms.append(transforms.Compose([
                transforms.ToPILImage(),
                transforms.AugMix(),
                transforms.ToTensor()
            ]))
        elif aug == "trivialaugment":
            # TrivialAugmentWide augmentation
            sample_transforms.append(transforms.Compose([
                transforms.ToPILImage(),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor()
            ]))
        elif aug in aug_map:
            sample_transforms.append(aug_map[aug])

    sample_transform = transforms.Compose(sample_transforms) if sample_transforms else None
    return {"use_mixup": use_mixup, "use_cutmix": use_cutmix, "sample_transform": sample_transform}







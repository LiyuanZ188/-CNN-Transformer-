import torch
from evaluation import evaluate_model
import os
from tqdm import tqdm
from augmentations import mixup_data, get_augmentation, cutmix_data
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



def train_model(model, dataloader, num_classes, augmentation_name, validation_loader=None, epochs=20, lr=0.001,
                optimizer_type="adam", weight_decay=0.0, momentum=0.9, patience=5, save_every=5,
                save_path=None):
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    # Setup optimizer
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    training_losses = []
    validation_losses = []
    validation_accuracies = []
    validation_precisions = []
    validation_recalls = []
    validation_f1_scores = []

    best_model_wts = None
    best_loss = float('inf')
    epochs_no_improve = 0

    # Get augmentation configuration.
    aug_config = get_augmentation(augmentation_name) if augmentation_name and augmentation_name.lower() != "none" else None
    sample_transform = aug_config["sample_transform"] if aug_config is not None else None
    use_mixup = aug_config["use_mixup"] if aug_config is not None else False
    use_cutmix = aug_config["use_cutmix"] if aug_config is not None else False

    # If sample-level augmentations are used, prepare transforms to invert normalization and reapply it after.
    if sample_transform is not None:
        normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
        inv_normalize_transform = transforms.Normalize(
            mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
            std=[1/s for s in [0.229, 0.224, 0.225]]
        )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for images, labels in progress_bar:
            if sample_transform is not None:
                # Ensure each image is on CPU and revert normalization to original pixel range before augmentation
                images = torch.stack([sample_transform(inv_normalize_transform(image.cpu())) for image in images])
                # Re-normalize augmented images for model input
                images = normalize_transform(images)
                images = images.to(device)
                labels = labels.to(device)
            else:
                images, labels = images.to(device), labels.to(device)

            # Apply batch-level augmentations if specified
            # Note: If both MixUp and CutMix are enabled, MixUp will be applied and CutMix will be ignored.
            if use_mixup:
                images, labels_a, labels_b, lam = mixup_data(images, labels)
                outputs = model(images)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            elif use_cutmix:
                images, labels_a, labels_b, lam = cutmix_data(images, labels)
                outputs = model(images)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            else:
                outputs = model(images)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(dataloader)
        training_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}")

        # Validation Phase
        if validation_loader:
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for images, labels in validation_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = criterion(logits, labels)
                    val_running_loss += loss.item()
            avg_val_loss = val_running_loss / len(validation_loader)
            validation_losses.append(avg_val_loss)
            print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")

            accuracy, precision, recall, f1 = evaluate_model(model, validation_loader, num_classes)
            validation_accuracies.append(accuracy)
            validation_precisions.append(precision)
            validation_recalls.append(recall)
            validation_f1_scores.append(f1)
            print(f"Validation Metrics after Epoch {epoch + 1}:\n"
                  f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model_wts = model.state_dict()
                epochs_no_improve = 0
                if save_path:
                    best_model_path = os.path.join(save_path, f"best_{num_classes}.pth")
                    torch.save(best_model_wts, best_model_path)
                    print(f"Best model saved at {best_model_path}")
            else:
                epochs_no_improve += 1

            if (epoch + 1) % save_every == 0 and save_path:
                checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best Validation Loss: {best_loss:.4f}")
            break

    if best_model_wts:
        print(f"Loading best model with Validation Loss: {best_loss:.4f}")
        model.load_state_dict(best_model_wts)

    if save_path:
        metrics_path = os.path.join(save_path, "training_metrics.pth")
        torch.save({
            "training_losses": training_losses,
            "validation_losses": validation_losses,
            "validation_accuracies": validation_accuracies,
            "validation_precisions": validation_precisions,
            "validation_recalls": validation_recalls,
            "validation_f1_scores": validation_f1_scores
        }, metrics_path)
        print(f"Training metrics saved at {metrics_path}")

    return model




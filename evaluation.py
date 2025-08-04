import torch
import torchmetrics

# Ensure model and metrics are on the same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def evaluate_model(model, dataloader, num_classes):
    model.eval()  # Set model to evaluation mode
    all_preds, all_labels = [], []

    #  Ensure torchmetrics are moved to the correct device
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    precision_metric = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro").to(device)
    recall_metric = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro").to(device)
    f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)  # Move to device

            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())  # Convert to numpy for processing
            all_labels.extend(labels.cpu().numpy())

    #  Move tensors back to device before passing to torchmetrics
    all_preds = torch.tensor(all_preds, device=device)
    all_labels = torch.tensor(all_labels, device=device)

    # Compute evaluation metrics
    accuracy = accuracy_metric(all_preds, all_labels).item()
    precision = precision_metric(all_preds, all_labels).item()
    recall = recall_metric(all_preds, all_labels).item()
    f1 = f1_metric(all_preds, all_labels).item()

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    return accuracy, precision, recall, f1




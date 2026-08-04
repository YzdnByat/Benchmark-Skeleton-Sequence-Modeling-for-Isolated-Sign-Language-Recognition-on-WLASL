# ==========================================
# SRC/EVALUATOR.PY - PLOTTING & CONFUSION MATRIX
# ==========================================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.amp import autocast
from src.utils import accuracy_top_k


def plot_training_history(history: dict, save_dir: str):
    """Plots and saves Train/Val Loss and Accuracy curves over epochs."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Plot
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-o', label='Val Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    # Accuracy Plot
    ax1_acc = ax2
    ax1_acc.plot(epochs, [a * 100 for a in history['train_acc']], 'b-o', label='Train Acc')
    ax1_acc.plot(epochs, [a * 100 for a in history['val_top1']], 'r-o', label='Val Top-1 Acc')
    ax1_acc.set_title('Training & Validation Accuracy (%)')
    ax1_acc.set_xlabel('Epochs')
    ax1_acc.set_ylabel('Accuracy (%)')
    ax1_acc.grid(True)
    ax1_acc.legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "loss_acc_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📊 Loss & Accuracy curves saved to: {plot_path}")


def evaluate_and_plot_cm(model, test_loader, exp_dir, device):
    """Runs test evaluation, saves Top-1/Top-5 scores, Confusion Matrix plot, and Classification Report."""
    print("\n" + "=" * 50)
    print("🏆 RUNNING TEST EVALUATION & GENERATING ANALYTICS")
    print("=" * 50)

    model.eval()
    all_preds = []
    all_targets = []
    test_top1_accs, test_top5_accs = [], []
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast(device_type):
                outputs = model(inputs)

            top1, top5 = accuracy_top_k(outputs, labels, topk=(1, 5))
            test_top1_accs.append(top1 * inputs.size(0))
            test_top5_accs.append(top5 * inputs.size(0))

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    test_size = len(test_loader.dataset)
    final_top1 = (sum(test_top1_accs) / test_size) * 100 if test_size > 0 else 0.0
    final_top5 = (sum(test_top5_accs) / test_size) * 100 if test_size > 0 else 0.0

    print(f"🌟 FINAL TEST TOP-1 ACCURACY: {final_top1:.2f}%")
    print(f"🌟 FINAL TEST TOP-5 ACCURACY: {final_top5:.2f}%")

    # 1. Save Confusion Matrix Plot
    if len(all_targets) > 0:
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, cmap="Blues", annot=False)
        plt.title(f"Confusion Matrix (Test Top-1 Acc: {final_top1:.2f}%)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        cm_path = os.path.join(exp_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"🖼️ Confusion Matrix heatmap saved to: {cm_path}")

        # 2. Save Full Classification Metrics Report to Text File
        report_path = os.path.join(exp_dir, "test_classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"FINAL TEST TOP-1 ACCURACY: {final_top1:.2f}%\n")
            f.write(f"FINAL TEST TOP-5 ACCURACY: {final_top5:.2f}%\n\n")
            f.write("CLASSIFICATION REPORT:\n")
            f.write(classification_report(all_targets, all_preds, zero_division=0))

        print(f"📄 Classification report saved to: {report_path}")
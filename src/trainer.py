# ==========================================
# SRC/TRAINER.PY - TRAINING STRATEGIES & LOOPS
# ==========================================
import copy
import os
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from src.utils import accuracy_top_k


def train_epoch(model, dataloader, criterion, optimizer, scaler, accumulation_steps, device):
    """Executes a single training epoch with AMP and Gradient Accumulation."""
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    optimizer.zero_grad()

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        with autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item() * accumulation_steps * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data).item()
        train_total += labels.size(0)

    return train_loss / train_total, train_correct / train_total


def evaluate(model, dataloader, device):
    """Evaluates model performance returning Top-1 and Top-5 accuracy."""
    model.eval()
    val_top1_accs, val_top5_accs = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast('cuda'):
                outputs = model(inputs)
            top1, top5 = accuracy_top_k(outputs, labels, topk=(1, 5))
            val_top1_accs.append(top1 * inputs.size(0))
            val_top5_accs.append(top5 * inputs.size(0))

    val_size = len(dataloader.dataset)
    return sum(val_top1_accs) / val_size, sum(val_top5_accs) / val_size


# ==========================================
# STRATEGY 1: EXACT PAPER END-TO-END
# ==========================================
def train_paper_strategy(model, train_loader, val_loader, config, exp_dir, device):
    """
    Implements the exact training regime from the Video Swin Transformer paper:
    - Backbone LR = 0.1x Head LR (Differential Learning Rates)
    - 2.5 Warmup Epochs followed by Cosine Annealing
    """
    paper_cfg = config["paper_config"]
    epochs = paper_cfg["epochs"]
    backbone_lr = paper_cfg["backbone_lr"]
    head_lr = paper_cfg["head_lr"]
    warmup_epochs = paper_cfg["warmup_epochs"]
    weight_decay = paper_cfg["weight_decay"]
    accumulation_steps = config.get("accumulation_steps", 2)

    print(f"\n🚀 Running Strategy: EXACT PAPER END-TO-END | Epochs: {epochs}")
    print(f"🔑 Backbone LR: {backbone_lr} | Head LR: {head_lr} | Warmup: {warmup_epochs} epochs")

    # Differential Learning Rate setup (Backbone vs Head)
    backbone_params = [p for n, p in model.named_parameters() if "head" not in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "head" in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr}
    ], weight_decay=weight_decay)

    # Cosine Annealing with Warmup Scheduler
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + torch.cos(torch.tensor((epoch - warmup_epochs) / (epochs - warmup_epochs) * 3.14159265)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.1))
    scaler = GradScaler('cuda')

    best_val_top1 = 0.0
    best_checkpoint_path = os.path.join(exp_dir, "best_paper_model.pth")

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, accumulation_steps,
                                            device)
        val_top1, val_top5 = evaluate(model, val_loader, device)
        scheduler.step()

        save_flag = ""
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save(model.state_dict(), best_checkpoint_path)
            save_flag = "💾 [SAVED]"

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | {time.time() - t0:.1f}s | Train Acc: {train_acc * 100:.2f}% | Val Top-1: {val_top1 * 100:.2f}% | Val Top-5: {val_top5 * 100:.2f}% {save_flag}")

    print(f"✅ Paper Strategy Complete! Best Val Top-1: {best_val_top1 * 100:.2f}%")
    return best_checkpoint_path


# ==========================================
# STRATEGY 2: PROGRESSIVE UNFREEZING
# ==========================================
def train_progressive_strategy(model, train_loader, val_loader, config, exp_dir, device):
    """Executes Progressive Unfreezing across 3 sequential phases."""
    prog_cfg = config["progressive_config"]
    accumulation_steps = config.get("accumulation_steps", 2)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.1))
    scaler = GradScaler('cuda')

    phases = [
        ("PHASE 1 (Head Only)", prog_cfg["epochs_phase1"], prog_cfg["lr_phase1"], "head"),
        ("PHASE 2 (Head + Stage 4)", prog_cfg["epochs_phase2"], prog_cfg["lr_phase2"], "stage4"),
        ("PHASE 3 (Full Fine-Tuning)", prog_cfg["epochs_phase3"], prog_cfg["lr_phase3"], "full")
    ]

    best_checkpoint_path = os.path.join(exp_dir, "best_progressive_model.pth")
    best_val_top1 = 0.0

    for phase_name, epochs, lr, unfreeze_target in phases:
        print(f"\n{'=' * 50}\n🚀 STARTING {phase_name} | LR: {lr} | Epochs: {epochs}\n{'=' * 50}")

        # Set parameter trainable flags
        if unfreeze_target == "head":
            for param in model.parameters(): param.requires_grad = False
            for param in model.head.parameters(): param.requires_grad = True
        elif unfreeze_target == "stage4":
            for param in model.norm.parameters(): param.requires_grad = True
            for param in model.features[-1].parameters(): param.requires_grad = True
        elif unfreeze_target == "full":
            for param in model.parameters(): param.requires_grad = True

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        for epoch in range(epochs):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, accumulation_steps,
                                                device)
            val_top1, val_top5 = evaluate(model, val_loader, device)
            scheduler.step()

            save_flag = ""
            if val_top1 > best_val_top1:
                best_val_top1 = val_top1
                torch.save(model.state_dict(), best_checkpoint_path)
                save_flag = "💾 [SAVED]"

            print(
                f"Epoch {epoch + 1:02d}/{epochs} | {time.time() - t0:.1f}s | Train Acc: {train_acc * 100:.2f}% | Val Top-1: {val_top1 * 100:.2f}% {save_flag}")

    print(f"✅ Progressive Strategy Complete! Best Val Top-1: {best_val_top1 * 100:.2f}%")
    return best_checkpoint_path
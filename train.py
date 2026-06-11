import time

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    seed,
    frozen_epochs,
    frozen_patience,
    ft_epochs,
    ft_patience,
    head_lr,
    encoder_lr,
    weight_decay,
    label_smooth,
    vit_checkpoint_path,
    vit_last_path,
    vit_ft_path,
    vit_ft_last_path,
)

from src.utils import set_seed, get_device
from src.data import create_train_val_loaders
from src.model import create_vit_model, unfreeze_last_two_blocks
from src.engine import train_one_epoch, evaluate, evaluate_with_tta


def train_frozen_vit(model, train_loader, val_loader, criterion, device, class_to_idx):
    #first part of training
    #ViT backbone is frozen, only the 100-class classifier head is trained

    optimizer = optim.AdamW(model.heads.head.parameters(), lr = head_lr, weight_decay = weight_decay)

    # ReduceLROnPlateau lowers the learning rate if validation accuracy stops improving.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor = 0.5,
        patience = 2
    )

    vit_history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    epochs_without_improvement = 0

    start_time = time.time()

    for epoch in range(frozen_epochs):
        epoch_num = epoch + 1

        print(f"\nViT Frozen Epoch {epoch_num}/{frozen_epochs}", flush = True)
        print("-" * 25, flush = True)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch_num = epoch_num
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        #decide whether to lower LR
        scheduler.step(val_acc)

        vit_history["train_loss"].append(train_loss)
        vit_history["train_acc"].append(train_acc)
        vit_history["val_loss"].append(val_loss)
        vit_history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} ; Train Acc: {train_acc:.4f}", flush = True)
        print(f"Val Loss:   {val_loss:.4f} ; Val Acc:   {val_acc:.4f}", flush = True)

        #save latest checkpoint each epoch in case training crashes
        torch.save({
            "epoch": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "history": vit_history,
            "class_to_idx": class_to_idx
        }, vit_last_path)

        #save best checkpoint when validation accuracy improves from previous best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "history": vit_history,
                "class_to_idx": class_to_idx
            }, vit_checkpoint_path)

            print(f"Saved new best ViT frozen model: {best_val_acc:.4f}", flush = True)

        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch.", flush = True)

        #early stop
        if epochs_without_improvement >= frozen_patience:
            print("Early stopping triggered.", flush = True)
            break

    end_time = time.time()

    print("\nFrozen ViT training complete.", flush = True)
    print(f"Best ViT frozen validation accuracy: {best_val_acc:.4f}", flush = True)
    print(f"Training time: {(end_time - start_time) / 60:.2f} minutes", flush = True)
    print("Best checkpoint:", vit_checkpoint_path)

    return model, best_val_acc


def fine_tune_vit(model, train_loader, val_loader, criterion, device, class_to_idx):
    #second part of training
    #uses best checkpoint
    #unfreezes classifier head and last two ViT encoder blocks

    checkpoint = torch.load(vit_checkpoint_path, map_location = device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print("Loaded best frozen ViT checkpoint.")
    print("Best frozen validation accuracy:", checkpoint["best_val_acc"])
    print("Saved at epoch:", checkpoint["epoch"])

    model = unfreeze_last_two_blocks(model)

    #last two encoder blocks use smaller lr than the classifier head
    fine_tune_encoder_params = []

    for block in model.encoder.layers[-2:]:
        fine_tune_encoder_params += list(block.parameters())

    optimizer = optim.AdamW(
        [
            {"params": model.heads.head.parameters(), "lr": head_lr},
            {"params": fine_tune_encoder_params, "lr": encoder_lr},
        ],
        weight_decay = weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = "max",
        factor = 0.5,
        patience = 2
    )

    vit_fine_tune_history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_finetune_val_acc = checkpoint["best_val_acc"]
    epochs_without_improvement = 0

    #creates fine-tuned checkpoint file, even if no improvement
    torch.save({
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_finetune_val_acc,
        "fine_tune_history": vit_fine_tune_history,
        "class_to_idx": class_to_idx
    }, vit_ft_path)

    print("Initialized fine-tuned checkpoint from frozen ViT.")
    print("Starting fine-tune val accuracy:", best_finetune_val_acc)

    start_time = time.time()

    for epoch in range(ft_epochs):
        epoch_num = epoch + 1

        print(f"\nViT Fine-tuning Epoch {epoch_num}/{ft_epochs}", flush = True)
        print("-" * 25, flush = True)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch_num=epoch_num
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_acc)

        vit_fine_tune_history["train_loss"].append(train_loss)
        vit_fine_tune_history["train_acc"].append(train_acc)
        vit_fine_tune_history["val_loss"].append(val_loss)
        vit_fine_tune_history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} ; Train Acc: {train_acc:.4f}", flush=True)
        print(f"Val Loss:   {val_loss:.4f} ; Val Acc:   {val_acc:.4f}", flush=True)

        # Save latest fine-tuned checkpoint every epoch.
        torch.save({
            "epoch": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "fine_tune_history": vit_fine_tune_history,
            "class_to_idx": class_to_idx
        }, vit_ft_last_path)

        # Save best fine-tuned checkpoint only when validation improves.
        if val_acc > best_finetune_val_acc:
            best_finetune_val_acc = val_acc
            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_finetune_val_acc,
                "fine_tune_history": vit_fine_tune_history,
                "class_to_idx": class_to_idx
            }, vit_ft_path)

            print(f"Saved new best ViT fine-tuned model: {best_finetune_val_acc:.4f}", flush = True)

        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch.", flush = True)

        if epochs_without_improvement >= ft_patience:
            print("Early stopping triggered.", flush = True)
            break

    end_time = time.time()

    print("\nViT fine-tuning complete.", flush=True)
    print(f"Best ViT fine-tuned validation accuracy: {best_finetune_val_acc:.4f}", flush=True)
    print(f"Fine-tuning time: {(end_time - start_time) / 60:.2f} minutes", flush=True)
    print("Best fine-tuned checkpoint:", vit_ft_path)

    return model, best_finetune_val_acc


def main():
    #main training pipeline

    set_seed(seed)

    device = get_device()
    print("Using device:", device)

    train_loader, val_loader, full_train_dataset = create_train_val_loaders()

    # During training, use pretrained=True so ViT starts from ImageNet weights.
    model = create_vit_model(device, pretrained = True)

    criterion = nn.CrossEntropyLoss(label_smoothing = label_smooth)

    model, frozen_acc = train_frozen_vit(
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        full_train_dataset.class_to_idx
    )

    model, finetune_acc = fine_tune_vit(
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        full_train_dataset.class_to_idx
    )

    # Optional final validation using TTA.
    tta_val_loss, tta_val_acc = evaluate_with_tta(
        model,
        val_loader,
        criterion,
        device
    )

    print("ViT TTA Validation Loss:", tta_val_loss)
    print("ViT TTA Validation Accuracy:", tta_val_acc)


if __name__ == "__main__":
    main()
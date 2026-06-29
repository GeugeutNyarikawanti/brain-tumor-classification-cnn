import torch
import torch.nn as nn
import torch.optim as optim

import settings.config as config

from Models.cnn import BrainTumorCNN
from Utils.data_loader import create_dataloaders
from Utils.trainer import train_one_epoch
from Utils.evaluator import validate_one_epoch


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def train():

    train_loader, test_loader = create_dataloaders()

    model = BrainTumorCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    best_val_acc = 0.0

    for epoch in range(config.NUM_EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        val_loss, val_acc = validate_one_epoch(
            model,
            test_loader,
            criterion,
            DEVICE
        )

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                "Models/best_model.pth"
            )

        print(
            f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    return {
        "model": model,
        "best_val_acc": best_val_acc,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies
    }


def main():

    history = train()

    print(
        f"\nBest Validation Accuracy : "
        f"{history['best_val_acc']:.4f}"
    )


if __name__ == "__main__":
    main()
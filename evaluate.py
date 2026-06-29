import torch

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import settings.config as config

from Models.cnn import BrainTumorCNN
from Utils.data_loader import create_dataloaders


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def evaluate():

    _, test_loader = create_dataloaders()

    model = BrainTumorCNN().to(DEVICE)

    model.load_state_dict(
        torch.load(
            "Models/best_model.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():

        for images, target in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

            labels.extend(
                target.numpy()
            )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    cm = confusion_matrix(
        labels,
        predictions
    )

    report = classification_report(
        labels,
        predictions,
        target_names=config.CLASS_NAMES
    )

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": predictions,
        "labels": labels
    }


def main():

    result = evaluate()

    print(
        f"Test Accuracy : {result['accuracy']:.4f}"
    )

    print()

    print(result["classification_report"])

    print()

    print(result["confusion_matrix"])


if __name__ == "__main__":
    main()
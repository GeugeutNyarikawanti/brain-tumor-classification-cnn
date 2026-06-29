import sys

import torch
from PIL import Image

import settings.config as config

from Models.cnn import BrainTumorCNN
from Utils.data_loader import get_test_transform
from Utils.predictor import predict


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    transform = get_test_transform()

    image = transform(image)

    model = BrainTumorCNN().to(DEVICE)

    model.load_state_dict(
        torch.load(
            "Models/best_model.pth",
            map_location=DEVICE
        )
    )

    prediction, confidence = predict(
        model,
        image,
        DEVICE
    )

    return {
        "prediction": config.CLASS_NAMES[prediction],
        "confidence": confidence
    }


def main():

    if len(sys.argv) != 2:

        print("Usage:")
        print("python predict.py <image_path>")
        return

    result = predict_image(sys.argv[1])

    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
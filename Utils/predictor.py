import torch


def predict(
    model,
    image,
    device
):
    model.eval()

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    return prediction.item(), confidence.item()
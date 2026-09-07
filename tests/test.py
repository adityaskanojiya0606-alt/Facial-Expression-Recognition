import torch

from src.model import EmotionCNN
from src.dataset import test_loader


# =====================================
# DEVICE
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =====================================
# LOAD FINAL V2 MODEL
# =====================================

model = EmotionCNN(
    num_classes=7
).to(device)


model.load_state_dict(
    torch.load(
        "models/best_emotion_cnn_v2.pth",
        map_location=device
    )
)


model.eval()

print("V2 model loaded successfully!")


# =====================================
# TEST MODEL
# =====================================

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)


        outputs = model(images)


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (
            predicted == labels
        ).sum().item()


# =====================================
# CALCULATE ACCURACY
# =====================================

accuracy = (
    100 * correct / total
)


print("\n==============================")

print(
    f"Final Test Accuracy: "
    f"{accuracy:.2f}%"
)

print("==============================")
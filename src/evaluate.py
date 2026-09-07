import torch
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

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
# EMOTION CLASSES
# =====================================

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


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
# PREDICTION
# =====================================

all_predictions = []
all_labels = []

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


        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


# =====================================
# ACCURACY
# =====================================

accuracy = 100 * correct / total

print(
    "\nTest Accuracy: "
    f"{accuracy:.2f}%"
)


# =====================================
# CLASSIFICATION REPORT
# =====================================

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=EMOTIONS,
        zero_division=0
    )
)


# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# =====================================
# SAVE CONFUSION MATRIX GRAPH
# =====================================

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=EMOTIONS,
    yticklabels=EMOTIONS
)

plt.xlabel("Predicted Emotion")

plt.ylabel("Actual Emotion")

plt.title(
    "Confusion Matrix - Facial Emotion Recognition V2"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix_v2.png",
    dpi=300
)

plt.close()


print(
    "\nConfusion matrix saved as "
    "confusion_matrix_v2.png"
)
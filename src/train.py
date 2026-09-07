import torch
import torch.nn as nn
import torch.optim as optim

from collections import Counter

from src.model import EmotionCNN
from src.dataset import train_loader, test_loader


# =====================================
# DEVICE
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =====================================
# CLASS WEIGHTS
# =====================================

class_counts = Counter(
    train_loader.dataset.targets
)

total_samples = len(
    train_loader.dataset
)

num_classes = 7

class_weights = []

for i in range(num_classes):

    weight = total_samples / (
        num_classes * class_counts[i]
    )

    class_weights.append(weight)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
).to(device)


# =====================================
# MODEL
# =====================================

model = EmotionCNN(
    num_classes=7
).to(device)


# =====================================
# LOSS FUNCTION
# =====================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# =====================================
# OPTIMIZER
# =====================================

optimizer = optim.AdamW(

    model.parameters(),

    lr=0.001,

    weight_decay=1e-4
)


# =====================================
# SCHEDULER
# =====================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=3
)


# =====================================
# TRAINING SETTINGS
# =====================================

epochs = 40

best_accuracy = 0.0

patience = 8

no_improvement = 0


# =====================================
# TRAINING LOOP
# =====================================

for epoch in range(epochs):

    # -----------------------------
    # TRAINING
    # -----------------------------

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += loss.item()


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss /
        len(train_loader)
    )


    train_accuracy = (
        100 * correct / total
    )


    # -----------------------------
    # EVALUATION
    # -----------------------------

    model.eval()

    test_correct = 0

    test_total = 0


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(images)


            _, predicted = torch.max(
                outputs,
                1
            )


            test_total += labels.size(0)


            test_correct += (
                predicted == labels
            ).sum().item()


    test_accuracy = (
        100 *
        test_correct /
        test_total
    )


    # -----------------------------
    # UPDATE LEARNING RATE
    # -----------------------------

    scheduler.step(
        test_accuracy
    )


    # -----------------------------
    # SAVE BEST MODEL
    # -----------------------------

    if test_accuracy > best_accuracy:

        best_accuracy = test_accuracy

        no_improvement = 0


        torch.save(

            model.state_dict(),

            "models/best_emotion_cnn_v2.pth"

        )


        print(
            "New best model saved!"
        )

    else:

        no_improvement += 1


    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------

    print(

        f"Epoch [{epoch + 1}/{epochs}] | "

        f"Loss: {train_loss:.4f} | "

        f"Train Accuracy: "
        f"{train_accuracy:.2f}% | "

        f"Test Accuracy: "
        f"{test_accuracy:.2f}%"

    )


    # -----------------------------
    # EARLY STOPPING
    # -----------------------------

    if no_improvement >= patience:

        print(
            "\nEarly stopping activated."
        )

        break


# =====================================
# FINAL RESULT
# =====================================

print("\nTraining completed!")

print(
    f"Best Test Accuracy: "
    f"{best_accuracy:.2f}%"
)
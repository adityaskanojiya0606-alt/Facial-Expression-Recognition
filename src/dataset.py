from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =====================================
# DATA PATHS
# =====================================

train_path = "data/fer2013/train"

test_path = "data/fer2013/test"


# =====================================
# TRAIN TRANSFORM
# =====================================

train_transform = transforms.Compose([

    transforms.Resize((48, 48)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

])


# =====================================
# TEST TRANSFORM
# =====================================

test_transform = transforms.Compose([

    transforms.Resize((48, 48)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

])


# =====================================
# TRAIN DATASET
# =====================================

train_dataset = datasets.ImageFolder(

    train_path,

    transform=train_transform

)


# =====================================
# TEST DATASET
# =====================================

test_dataset = datasets.ImageFolder(

    test_path,

    transform=test_transform

)


# =====================================
# DATALOADERS
# =====================================

train_loader = DataLoader(

    train_dataset,

    batch_size=32,

    shuffle=True

)


test_loader = DataLoader(

    test_dataset,

    batch_size=32,

    shuffle=False

)


# =====================================
# DATASET INFORMATION
# =====================================

print("=====================================")

print("DATASET INFORMATION")

print("=====================================")

print("Training Images:", len(train_dataset))

print("Test Images:", len(test_dataset))

print("Classes:", train_dataset.classes)

print("Class Mapping:", train_dataset.class_to_idx)


# =====================================
# CHECK SAMPLE IMAGE
# =====================================

image, label = train_dataset[0]

print("\nSample Image Shape:", image.shape)

print("Sample Label:", label)


# =====================================
# CHECK BATCH
# =====================================

images, labels = next(iter(train_loader))

print("\nBatch Image Shape:", images.shape)

print("Batch Label Shape:", labels.shape)

print("=====================================")
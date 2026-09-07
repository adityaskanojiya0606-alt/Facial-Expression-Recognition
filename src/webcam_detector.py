import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.model import EmotionCNN


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
# DEVICE
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =====================================
# LOAD V2 MODEL
# =====================================

model = EmotionCNN(num_classes=7).to(device)

model.load_state_dict(
    torch.load(
        "models/best_emotion_cnn_v2.pth",
        map_location=device
    )
)

model.eval()

print("V2 model loaded successfully!")


# =====================================
# FACE DETECTOR
# =====================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# =====================================
# IMAGE PREPROCESSING
# MUST MATCH TRAINING PREPROCESSING
# =====================================

transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize((48, 48)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

])


# =====================================
# START WEBCAM
# =====================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Could not access webcam")
    exit()


print("Webcam started.")
print("Press Q to quit.")


# =====================================
# WEBCAM LOOP
# =====================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Failed to capture webcam frame")
        break


    # -----------------------------
    # CONVERT TO GRAYSCALE
    # Only for face detection
    # -----------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # -----------------------------
    # DETECT FACE
    # -----------------------------

    faces = face_cascade.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=5,

        minSize=(50, 50)

    )


    # -----------------------------
    # PROCESS EACH FACE
    # -----------------------------

    for (x, y, w, h) in faces:


        # Add small padding around face

        padding = int(0.15 * w)


        x1 = max(0, x - padding)

        y1 = max(0, y - padding)

        x2 = min(
            frame.shape[1],
            x + w + padding
        )

        y2 = min(
            frame.shape[0],
            y + h + padding
        )


        # -----------------------------
        # CROP FACE FROM COLOR FRAME
        # -----------------------------

        face = frame[y1:y2, x1:x2]


        # Safety check

        if face.size == 0:

            continue


        # -----------------------------
        # BGR TO RGB
        # -----------------------------

        face = cv2.cvtColor(

            face,

            cv2.COLOR_BGR2RGB

        )


        # -----------------------------
        # PREPROCESS FACE
        # -----------------------------

        face_tensor = transform(face)


        # Add batch dimension

        face_tensor = face_tensor.unsqueeze(0)


        # Move to device

        face_tensor = face_tensor.to(device)


        # -----------------------------
        # MODEL PREDICTION
        # -----------------------------

        with torch.no_grad():

            outputs = model(face_tensor)


            probabilities = F.softmax(

                outputs,

                dim=1

            )


            confidence, predicted = torch.max(

                probabilities,

                1

            )


        # -----------------------------
        # GET RESULT
        # -----------------------------

        emotion = EMOTIONS[
            predicted.item()
        ]


        confidence_percentage = (

            confidence.item() * 100

        )


        # -----------------------------
        # DRAW FACE RECTANGLE
        # -----------------------------

        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            (0, 255, 0),

            2

        )


        # -----------------------------
        # DISPLAY EMOTION
        # -----------------------------

        text = (

            f"{emotion}: "

            f"{confidence_percentage:.1f}%"

        )


        cv2.putText(

            frame,

            text,

            (x, y - 10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2

        )


    # =====================================
    # SHOW WEBCAM
    # =====================================

    cv2.imshow(

        "Facial Emotion Recognition - V2",

        frame

    )


    # =====================================
    # PRESS Q TO QUIT
    # =====================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# =====================================
# RELEASE RESOURCES
# =====================================

cap.release()

cv2.destroyAllWindows()

print("Program closed")
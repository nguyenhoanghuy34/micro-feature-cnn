import torch
from PIL import Image
from torchvision import transforms

from model import SkinCancerCNN


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


CLASSES = [
    "benign",
    "malignant"
]


transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# Model
# =========================================================

model = SkinCancerCNN(
    num_classes=2
)

model.load_state_dict(
    torch.load(
        "../checkpoints/best_model.pth",
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()


# =========================================================
# Image
# =========================================================

image = Image.open(
    "../example.jpg"
).convert("RGB")


image = transform(image)

image = image.unsqueeze(0)

image = image.to(DEVICE)


# =========================================================
# Prediction
# =========================================================

with torch.no_grad():

    logits = model(image)

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    prediction = probabilities.argmax(
        dim=1
    ).item()


print(
    "Prediction:",
    CLASSES[prediction]
)

print(
    "Probability:",
    probabilities[0][prediction].item()
)
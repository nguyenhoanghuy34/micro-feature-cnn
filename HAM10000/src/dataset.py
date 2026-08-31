from torchvision import datasets, transforms
from torch.utils.data import DataLoader


IMAGE_SIZE = 224
BATCH_SIZE = 32


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def create_dataloaders(data_dir):

    train_dataset = datasets.ImageFolder(
        f"{data_dir}/train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        f"{data_dir}/val",
        transform=val_test_transform
    )

    test_dataset = datasets.ImageFolder(
        f"{data_dir}/test",
        transform=val_test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.classes
    )
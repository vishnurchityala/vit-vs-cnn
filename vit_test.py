import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from src.model import PreTrainedViTClassifier
from src.train import PyTorchTrainer


def get_dataloaders(dataset_name="cifar10", batch_size=64, img_size=224, val_split=0.1):
    """
    Returns train_loader, val_loader, test_loader for CIFAR-10 or MNIST.
    Splits the official train set into (train + val), keeps test set separate.
    """
    if dataset_name.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # normalize RGB
        ])

        full_train_set = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        num_classes = 10

    elif dataset_name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),  # ViT needs 3-channel
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # works fine with 3 channels
        ])

        full_train_set = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

        num_classes = 10

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Split train into train/val
    val_size = int(len(full_train_set) * val_split)
    train_size = len(full_train_set) - val_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, num_classes


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Choose dataset: "cifar10" or "mnist"
    print("Loading the dataset......")
    dataset = "cifar10"  
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        dataset, batch_size=64, img_size=224, val_split=0.1
    )

    # Model
    print("Loading the model......")
    pretrained_cls = PreTrainedViTClassifier(
        num_classes=num_classes, model_name="vit_b_16", img_size=224, device=device
    )

    # Trainer
    print("Loading the trainer......")
    trainer = PyTorchTrainer(
        model=pretrained_cls,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        mixed_precision=True,
    )

    print("Training the model......")
    # Train for a few epochs
    history = trainer.fit(epochs=5)

    print("Evaluating the model.....")
    # (Optional) Evaluate on test set
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

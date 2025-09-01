import torch
from src.model import CustomViTClassifier, ResNetMLP
from src.train import PyTorchTrainer
from src.data_loaders import get_dataloaders

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
    pretrained_cls = CustomViTClassifier(
        num_classes=num_classes, model_name="vit_b_16", img_size=224, device=device
    )
    cnn_cls = ResNetMLP(
        num_classes=num_classes
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
    history = trainer.fit(epochs=1)

    print("Evaluating the model.....")
    # (Optional) Evaluate on test set
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    print("Saving the model...")
    torch.save(pretrained_cls.state_dict(), "model.pth")


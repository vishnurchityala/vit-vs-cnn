import torch
from src.model import CustomViTClassifier, ResNetMLP
from src.train import PyTorchTrainer
from src.data_loaders import get_dataloaders

if __name__ == "__main__":

    print("Loading the CIFAR Dataset......")
    dataset = "cifar10"  
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        dataset, batch_size=64, img_size=224, val_split=0.1
    )

    print("Loading the CNN Model......")
    cnn_cls = ResNetMLP(
        num_classes=num_classes
    )
    # Model
    print("Loading the ViT Model......")
    vit_cls = CustomViTClassifier(
        num_classes=num_classes, model_name="vit_b_16", img_size=224
    )


    # Trainer
    print("Loading the CNN Trainer......")
    cnn_trainer = PyTorchTrainer(
        model=cnn_cls,
        train_loader=train_loader,
        val_loader=val_loader,
        mixed_precision=True,
    )
    print("Loading the ViT Trainer......")
    vit_trainer = PyTorchTrainer(
        model=vit_cls,
        train_loader=train_loader,
        val_loader=val_loader,
        mixed_precision=True,
    )


    print("Training the CNN Model......")
    # Train for a few epochs
    history = cnn_trainer.fit(epochs=1)

    print("Evaluating the CNN Model.....")
    # (Optional) Evaluate on test set
    test_loss, test_acc = cnn_trainer.evaluate(test_loader)
    print(f"Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    print("Saving the CNN Model...")
    torch.save(cnn_cls.state_dict(), "cnn_model.pth")

    print("Training the ViT Model......")
    # Train for a few epochs
    history = vit_trainer.fit(epochs=1)

    print("Evaluating the ViT Model.....")
    # (Optional) Evaluate on test set
    test_loss, test_acc = vit_trainer.evaluate(test_loader)
    print(f"Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    print("Saving the ViT Model...")
    torch.save(vit_cls.state_dict(), "vit_model.pth")


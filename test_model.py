import torch
from src.model import CustomViTClassifier, ResNetMLP
from src.train import PyTorchTrainer
from src.data_loaders import dtd_test_loader, dtd_train_loader, dtd_val_loader, dtd_num_classes


def main():
    """Main training and evaluation script."""
    
    print("Loading the DTD Dataset...")
    train_loader = dtd_train_loader
    val_loader = dtd_val_loader
    test_loader = dtd_test_loader
    num_classes = dtd_num_classes
    
    print(f"Dataset loaded: {num_classes} classes")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create CNN model
    print("\nLoading the CNN Model...")
    cnn_model = ResNetMLP(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=False
    )
    
    # Create ViT model
    print("\nLoading the ViT Model...")
    vit_model = CustomViTClassifier(
        num_classes=num_classes,
        model_name="vit_b_16",
        img_size=224,
        pretrained=True,
        freeze_backbone=False
    )
    
    # Create CNN trainer
    print("\nLoading the CNN Trainer...")
    cnn_trainer = PyTorchTrainer(
        model=cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_amp=True,  # Enable mixed precision if CUDA available
        lr=1e-4
    )
    
    # Create ViT trainer
    print("\nLoading the ViT Trainer...")
    vit_trainer = PyTorchTrainer(
        model=vit_model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_amp=True,  # Enable mixed precision if CUDA available
        lr=1e-4
    )
    
    # Print model information
    print("\n" + "="*50)
    print("CNN Model Information:")
    cnn_trainer.get_model_info()
    
    print("\n" + "="*50)
    print("ViT Model Information:")
    vit_trainer.get_model_info()
    
    # Train ViT model
    print("\n" + "="*50)
    print("Training the ViT Model...")
    vit_history = vit_trainer.fit(epochs=2)
    
    # Evaluate ViT model
    print("\nEvaluating the ViT Model...")
    vit_test_loss, vit_test_acc = vit_trainer.evaluate(test_loader)
    print(f"ViT Final Test Loss: {vit_test_loss:.4f}, Test Acc: {vit_test_acc:.2f}%")
    
    # Save ViT model
    print("\nSaving the ViT Model...")
    vit_trainer.save_model("vit_model.pth", epoch=2)
    
    # Train CNN model
    print("\n" + "="*50)
    print("Training the CNN Model...")
    cnn_history = cnn_trainer.fit(epochs=2)
    
    # Evaluate CNN model
    print("\nEvaluating the CNN Model...")
    cnn_test_loss, cnn_test_acc = cnn_trainer.evaluate(test_loader)
    print(f"CNN Final Test Loss: {cnn_test_loss:.4f}, Test Acc: {cnn_test_acc:.2f}%")
    
    # Save CNN model
    print("\nSaving the CNN Model...")
    cnn_trainer.save_model("cnn_model.pth", epoch=2)
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"ViT Test Accuracy: {vit_test_acc:.2f}%")
    print(f"CNN Test Accuracy: {cnn_test_acc:.2f}%")
    
    if vit_test_acc > cnn_test_acc:
        print("ViT performed better!")
    elif cnn_test_acc > vit_test_acc:
        print("CNN performed better!")
    else:
        print("Both models performed equally!")
    
    print("="*70)


if __name__ == "__main__":
    main()
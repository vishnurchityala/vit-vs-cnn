import torch
from src.model import ResNetMLP,EfficientNetMLP
from src.train import PyTorchTrainerCuda
from src.data_loaders import dtd_test_loader, dtd_train_loader, dtd_val_loader, dtd_num_classes


def main():
    """Test and train the CNN (ResNet) model."""
    
    print("=" * 60)
    print("CNN MODEL TESTING")
    print("=" * 60)
    
    # Load dataset
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
    print("\nCreating the CNN Model...")
    cnn_model = EfficientNetMLP(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=False
    )
    
    # Create CNN trainer
    print("\nCreating the CNN Trainer...")
    cnn_trainer = PyTorchTrainerCuda(
        model=cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_amp=True,  # Enable mixed precision if CUDA available
        lr=1e-4,
        weight_decay=0.01
    )
    
    # Print model information
    print("\n" + "="*60)
    print("CNN MODEL INFORMATION")
    print("="*60)
    cnn_trainer.get_model_info()
    
    # Train CNN model
    print("\n" + "="*60)
    print("TRAINING CNN MODEL")
    print("="*60)
    cnn_history = cnn_trainer.fit(epochs=50)
    
    # Evaluate CNN model
    print("\n" + "="*60)
    print("EVALUATING CNN MODEL")
    print("="*60)
    cnn_test_loss, cnn_test_acc = cnn_trainer.evaluate(test_loader)
    
    # Save CNN model
    print("\nSaving the CNN Model...")
    cnn_trainer.save_model("cnn_model.pth", epoch=5)
    
    # Final results
    print("\n" + "="*60)
    print("CNN TRAINING RESULTS")
    print("="*60)
    print(f"Final Test Loss: {cnn_test_loss:.4f}")
    print(f"Final Test Accuracy: {cnn_test_acc:.2f}%")
    
    print(f"\nTraining History:")
    print(f"  Train Loss: {cnn_history['train_loss'][-1]:.4f}")
    print(f"  Train Accuracy: {cnn_history['train_acc'][-1]:.2f}%")
    print(f"  Val Loss: {cnn_history['val_loss'][-1]:.4f}")
    print(f"  Val Accuracy: {cnn_history['val_acc'][-1]:.2f}%")
    
    print("="*60)
    print("CNN TESTING COMPLETED!")
    print("="*60)
    
    return cnn_test_loss, cnn_test_acc


if __name__ == "__main__":
    main()
import torch
from src.model import ConvNeXtXXL_MLP
from src.train import PyTorchTrainerCuda, PyTorchTrainer
from src.data_loaders import dtd_test_loader, dtd_train_loader, dtd_val_loader, dtd_num_classes


def main():
    """Test and train the CNN (ConvNeXtXXL) model."""
    
    print("=" * 60)
    print("CNN (ConvNeXtXXL) MODEL TESTING")
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
    
    # Create CNN model with improved regularization
    print("\nCreating the CNN Model...")
    cnn_model = ConvNeXtXXL_MLP(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=True,
        mlp_hidden=[1024, 512, 256]
    )
    
    # Create CNN trainer with improved regularization settings
    print("\nCreating the CNN Trainer...")
    cnn_trainer = PyTorchTrainer(
        model=cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_amp=True,  # Enable mixed precision if CUDA available
        lr=3e-5,                        # Reduced from 1e-4 for better stability
        weight_decay=0.03,              # Increased from 0.01 for more regularization
        early_stopping_patience=10,     # Reduced from 15 for faster stopping
        early_stopping_min_delta=0.02,  # More strict improvement threshold
        lr_scheduler_patience=5         # Reduced from 7 for faster LR decay
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
    cnn_history = cnn_trainer.fit(epochs=30)  # Reduced from 50 to prevent overfitting
    
    # Evaluate CNN model
    print("\n" + "="*60)
    print("EVALUATING CNN MODEL")
    print("="*60)
    cnn_test_loss, cnn_test_acc = cnn_trainer.evaluate(test_loader)
    
    # Save CNN model
    print("\nSaving the CNN Model...")
    torch.save(cnn_model.state_dict(),"convnext_model.pth")
    
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
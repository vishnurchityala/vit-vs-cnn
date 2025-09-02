import torch
from src.model import CustomViTClassifier
from src.train import PyTorchTrainer
from src.data_loaders import dtd_test_loader, dtd_train_loader, dtd_val_loader, dtd_num_classes


def main():
    """Test and train the ViT (Vision Transformer) model."""
    
    print("=" * 60)
    print("VIT MODEL TESTING")
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
    
    # Create ViT model
    print("\nCreating the ViT Model...")
    vit_model = CustomViTClassifier(
        num_classes=num_classes,
        model_name="vit_b_16",
        img_size=224,
        pretrained=True,
        freeze_backbone=False
    )
    
    # Create ViT trainer
    print("\nCreating the ViT Trainer...")
    vit_trainer = PyTorchTrainer(
        model=vit_model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_amp=True,  # Enable mixed precision if CUDA available
        lr=1e-4,
        weight_decay=0.01
    )
    
    # Print model information
    print("\n" + "="*60)
    print("VIT MODEL INFORMATION")
    print("="*60)
    vit_trainer.get_model_info()
    
    # Train ViT model
    print("\n" + "="*60)
    print("TRAINING VIT MODEL")
    print("="*60)
    vit_history = vit_trainer.fit(epochs=5)
    
    # Evaluate ViT model
    print("\n" + "="*60)
    print("EVALUATING VIT MODEL")
    print("="*60)
    vit_test_loss, vit_test_acc = vit_trainer.evaluate(test_loader)
    
    # Save ViT model
    print("\nSaving the ViT Model...")
    vit_trainer.save_model("vit_model.pth", epoch=5)
    
    # Final results
    print("\n" + "="*60)
    print("VIT TRAINING RESULTS")
    print("="*60)
    print(f"Final Test Loss: {vit_test_loss:.4f}")
    print(f"Final Test Accuracy: {vit_test_acc:.2f}%")
    
    print(f"\nTraining History:")
    print(f"  Train Loss: {vit_history['train_loss'][-1]:.4f}")
    print(f"  Train Accuracy: {vit_history['train_acc'][-1]:.2f}%")
    print(f"  Val Loss: {vit_history['val_loss'][-1]:.4f}")
    print(f"  Val Accuracy: {vit_history['val_acc'][-1]:.2f}%")
    
    print("="*60)
    print("VIT TESTING COMPLETED!")
    print("="*60)
    
    return vit_test_loss, vit_test_acc


if __name__ == "__main__":
    main()

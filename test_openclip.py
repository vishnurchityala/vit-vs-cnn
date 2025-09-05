import torch
from src.model import OpenCLIPViTL14_MLP
from src.train import PyTorchTrainerCuda, PyTorchTrainer
from src.data_loaders import dtd_test_loader, dtd_train_loader, dtd_val_loader, dtd_num_classes


def main():
    """Test and train the OpenCLIP ViT-L/14 model."""
    
    print("=" * 60)
    print("OPENCLIP ViT-L/14 MODEL TESTING")
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
    
    # Create OpenCLIP model
    print("\nCreating the OpenCLIP ViT-L/14 Model...")
    openclip_model = OpenCLIPViTL14_MLP(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=True,
        mlp_hidden=[1024, 512, 256]
    )
    
    # Create OpenCLIP trainer
    print("\nCreating the OpenCLIP Trainer...")
    openclip_trainer = PyTorchTrainer(
        model=openclip_model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_amp=True,
        lr=3e-5,
        weight_decay=0.03,
        early_stopping_patience=10,
        early_stopping_min_delta=0.02,
        lr_scheduler_patience=5
    )
    
    # Print model information
    print("\n" + "="*60)
    print("OPENCLIP MODEL INFORMATION")
    print("="*60)
    openclip_trainer.get_model_info()
    
    # Train OpenCLIP model
    print("\n" + "="*60)
    print("TRAINING OPENCLIP MODEL")
    print("="*60)
    openclip_history = openclip_trainer.fit(epochs=50)
    
    # Evaluate OpenCLIP model
    print("\n" + "="*60)
    print("EVALUATING OPENCLIP MODEL")
    print("="*60)
    openclip_test_loss, openclip_test_acc = openclip_trainer.evaluate(test_loader)
    
    # Save OpenCLIP model
    print("\nSaving the OpenCLIP Model...")
    torch.save(openclip_model.state_dict(), "openclip_vit_l14_model.pth")
    
    # Final results
    print("\n" + "="*60)
    print("OPENCLIP TRAINING RESULTS")
    print("="*60)
    print(f"Final Test Loss: {openclip_test_loss:.4f}")
    print(f"Final Test Accuracy: {openclip_test_acc:.2f}%")
    
    print(f"\nTraining History:")
    print(f"  Train Loss: {openclip_history['train_loss'][-1]:.4f}")
    print(f"  Train Accuracy: {openclip_history['train_acc'][-1]:.2f}%")
    print(f"  Val Loss: {openclip_history['val_loss'][-1]:.4f}")
    print(f"  Val Accuracy: {openclip_history['val_acc'][-1]:.2f}%")
    
    print("="*60)
    print("OPENCLIP TESTING COMPLETED!")
    print("="*60)
    
    return openclip_test_loss, openclip_test_acc


if __name__ == "__main__":
    main()

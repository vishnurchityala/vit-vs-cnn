#!/usr/bin/env python3
"""
Test script to verify graceful CUDA device handling
This should work on Mac (development) and Windows server (training)
"""

import torch
from src.model import CustomViTClassifier, ResNetMLP
from src.train import PyTorchTrainer

def test_device_handling():
    print("Testing Graceful Device Handling")
    print("=" * 50)
    
    # Test 1: Model Creation
    print("\n1. Testing Model Creation...")
    try:
        # Test ResNet model
        print("   Creating ResNet model...")
        cnn_model = ResNetMLP(num_classes=10)
        print(f"   ResNet created successfully on {cnn_model.device}")
        
        # Test ViT model
        print("   Creating ViT model...")
        vit_model = CustomViTClassifier(num_classes=10)
        print(f"   ViT created successfully on {vit_model.device}")
        
    except Exception as e:
        print(f"   ERROR: Model creation failed: {e}")
        return False
    
    # Test 2: Trainer Creation
    print("\n2. Testing Trainer Creation...")
    try:
        # Test CNN trainer
        print("   Creating CNN trainer...")
        cnn_trainer = PyTorchTrainer(model=cnn_model, mixed_precision=True)
        print(f"   CNN trainer created on {cnn_trainer.device}")
        
        # Test ViT trainer
        print("   Creating ViT trainer...")
        vit_trainer = PyTorchTrainer(model=vit_model, mixed_precision=True)
        print(f"   ViT trainer created on {vit_trainer.device}")
        
    except Exception as e:
        print(f"   ERROR: Trainer creation failed: {e}")
        return False
    
    # Test 3: Forward Pass
    print("\n3. Testing Forward Pass...")
    try:
        # Create dummy data
        dummy_input = torch.randn(2, 3, 224, 224)
        
        # Test CNN forward pass
        print("   Testing CNN forward pass...")
        cnn_output = cnn_model(dummy_input)
        print(f"   CNN forward pass successful, output shape: {cnn_output.shape}")
        
        # Test ViT forward pass
        print("   Testing ViT forward pass...")
        vit_output = vit_model(dummy_input)
        print(f"   ViT forward pass successful, output shape: {vit_output.shape}")
        
    except Exception as e:
        print(f"   ERROR: Forward pass failed: {e}")
        return False
    
    # Test 4: Device Detection
    print("\n4. Testing Device Detection...")
    try:
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA Device Count: {torch.cuda.device_count()}")
            print(f"   Current CUDA Device: {torch.cuda.current_device()}")
        
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   Default Device Type: {torch.tensor([1.0]).device}")
        
    except Exception as e:
        print(f"   WARNING: Device detection warning: {e}")
    
    print("\n" + "=" * 50)
    print("All tests passed! Device handling is working correctly.")
    print("The code should work on both Mac (development) and Windows server (training).")
    return True

if __name__ == "__main__":
    test_device_handling()

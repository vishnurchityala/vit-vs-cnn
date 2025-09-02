import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np


def get_device():
    """Determine the appropriate device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class PyTorchTrainer:
    """
    Clean PyTorch trainer with proper device handling.
    Works on Mac (development) and Windows server (training).
    """
    
    def __init__(self, 
                 model,
                 train_loader=None,
                 val_loader=None,
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 device=None,
                 use_amp=False,
                 lr=3e-4,
                 weight_decay=0.05):
        
        # Device handling
        self.device = device if device is not None else get_device()
        print(f"[INFO] Using device: {self.device}")
        
        # Print device info
        if self.device.type == "cuda":
            print(f"[INFO] CUDA Device: {torch.cuda.get_device_name()}")
            print(f"[INFO] CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("[INFO] Using CPU")
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Training components
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = scheduler
        
        # Mixed precision setup
        self.use_amp = use_amp and self.device.type == "cuda"
        if self.use_amp:
            self.scaler = GradScaler()
            print("[INFO] Mixed precision training enabled")
        else:
            self.scaler = None
            if use_amp and self.device.type == "cpu":
                print("[INFO] Mixed precision disabled on CPU")
            else:
                print("[INFO] Mixed precision disabled")
    
    def train_one_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if self.use_amp:
                with autocast(device_type="cuda"):
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            
            # Update scheduler if provided
            if self.scheduler:
                self.scheduler.step()
            
            # Calculate metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            accuracy = 100.0 * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        return total_loss / len(self.train_loader), 100.0 * correct / total
    
    def validate(self, epoch):
        """Validate the model."""
        if self.val_loader is None:
            return None, None
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
            
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast(device_type="cuda"):
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Calculate metrics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Update progress bar
                accuracy = 100.0 * correct / total
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })
        
        return total_loss / len(self.val_loader), 100.0 * correct / total
    
    def fit(self, epochs=10):
        """Train the model for specified number of epochs."""
        if self.train_loader is None:
            raise ValueError("train_loader must be provided for training")
        
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        print(f"\nStarting training for {epochs} epochs...")
        print("=" * 70)
        
        for epoch in range(1, epochs + 1):
            # Training
            train_loss, train_acc = self.train_one_epoch(epoch)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate(epoch)
            if val_loss is not None:
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
            
            # Print epoch results
            if val_loss is not None:
                print(f"Epoch [{epoch:3d}/{epochs}] "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
                      f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            else:
                print(f"Epoch [{epoch:3d}/{epochs}] "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        print("=" * 70)
        print("Training completed!")
        return history
    
    def evaluate(self, test_loader, verbose=True):
        """Evaluate the model on test dataset."""
        if test_loader is None:
            raise ValueError("test_loader must be provided for evaluation")
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            if verbose:
                pbar = tqdm(test_loader, desc="Evaluating")
                iterator = pbar
            else:
                iterator = test_loader
            
            for batch_idx, (data, target) in enumerate(iterator):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast(device_type="cuda"):
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Calculate metrics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                if verbose:
                    accuracy = 100.0 * correct / total
                    avg_loss = total_loss / (batch_idx + 1)
                    pbar.set_postfix({
                        'Loss': f'{avg_loss:.4f}',
                        'Acc': f'{accuracy:.2f}%'
                    })
        
        test_loss = total_loss / len(test_loader)
        test_acc = 100.0 * correct / total
        
        if verbose:
            print("=" * 50)
            print("Test Results:")
            print(f"   Test Loss: {test_loss:.4f}")
            print(f"   Test Accuracy: {test_acc:.2f}%")
            print(f"   Correct Predictions: {correct}/{total}")
            print("=" * 50)
        
        return test_loss, test_acc
    
    def predict(self, data_loader, return_probabilities=False):
        """Make predictions on a dataset."""
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc="Predicting")
            for data, _ in pbar:
                data = data.to(self.device)
                
                if self.use_amp:
                    with autocast(device_type="cuda"):
                        output = self.model(data)
                else:
                    output = self.model(data)
                
                if return_probabilities:
                    probs = torch.softmax(output, dim=1)
                    all_predictions.extend(probs.cpu().numpy())
                else:
                    pred = output.argmax(dim=1)
                    all_predictions.extend(pred.cpu().numpy())
        
        return np.array(all_predictions)
    
    def save_model(self, path, epoch=None, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'use_amp': self.use_amp,
            **kwargs
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        print(f"Model loaded from {path} (epoch {epoch})")
        return epoch
    
    def get_model_info(self):
        """Print model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("=" * 60)
        print("Model Information:")
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Trainable Parameters: {trainable_params:,}")
        print(f"   Non-trainable Parameters: {total_params - trainable_params:,}")
        print(f"   Device: {self.device}")
        print(f"   Mixed Precision: {self.use_amp}")
        print(f"   Model Type: {type(self.model).__name__}")
        
        # Calculate model size in MB
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        print(f"   Model Size: {model_size:.2f} MB")
        print("=" * 60)
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp  # Fixed import - use torch.amp instead of torch.cuda.amp
from tqdm import tqdm


class PyTorchTrainer:
    def __init__(self, model,
                 train_loader=None,
                 val_loader=None,
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 device=None,
                 mixed_precision=False,
                 lr=3e-4,
                 weight_decay=0.05):
        """
        Generic training loop wrapper for PyTorch models with verbosity,
        automatic device detection, and mixed precision support.
        """
        # Smart device detection with MPS support
        self.device = self._detect_device(device)
        self.model = model.to(self.device)

        # Set device_type for autocast - critical for proper AMP functionality
        if self.device.startswith("cuda"):
            self.device_type = "cuda"
        elif self.device == "mps":
            self.device_type = "cpu"  # MPS uses cpu device_type for autocast
        else:
            self.device_type = "cpu"

        # Verbosity with better device info
        print("=" * 60)
        if self.device == "cuda":
            print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif self.device == "mps":
            print("[INFO] Using Apple Silicon GPU (MPS)")
        else:
            print("[INFO] Using CPU")
        print("=" * 60)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training components with better defaults
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = scheduler

        # Mixed precision setup with device-specific handling
        self.mixed_precision = self._setup_mixed_precision(mixed_precision)
        
        # Initialize scaler with proper device_type
        if self.device == "cuda" and self.mixed_precision:
            self.scaler = amp.GradScaler("cuda", enabled=True)
            self.autocast_dtype = torch.float16
        else:
            self.scaler = amp.GradScaler("cpu", enabled=False)
            self.autocast_dtype = torch.float32
            
        # Print mixed precision status
        if self.mixed_precision:
            print("[INFO] Mixed precision training enabled.")
        else:
            print("[INFO] Mixed precision training disabled.")
            if self.device == "mps":
                print("[INFO] Mixed precision not supported on MPS, using float32.")

    def _detect_device(self, device=None):
        """Smart device detection with priority: CUDA > MPS > CPU"""
        if device is not None:
            return device
            
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _setup_mixed_precision(self, mixed_precision):
        """Setup mixed precision based on device capabilities"""
        if not mixed_precision:
            return False
            
        if self.device == "cuda":
            return True
        elif self.device == "mps":
            # MPS has issues with mixed precision, disable it
            print("[WARNING] Mixed precision disabled on MPS due to compatibility issues.")
            return False
        else:
            # CPU mixed precision is generally not beneficial
            return False

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for images, labels in pbar:
            # Non-blocking transfer for better performance
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # More efficient gradient zeroing
            self.optimizer.zero_grad(set_to_none=True)

            # Fixed autocast with proper device_type and dtype handling
            with amp.autocast(
                device_type=self.device_type, 
                dtype=self.autocast_dtype, 
                enabled=self.mixed_precision
            ):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Proper backward pass handling for different devices
            if self.mixed_precision and self.device == "cuda":
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Step scheduler if it's step-based
            if self.scheduler:
                self.scheduler.step()

            # Metrics calculation
            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total_correct += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

            # Update progress bar
            current_acc = total_correct / total_samples
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.4f}")

        return total_loss / total_samples, total_correct / total_samples

    def validate(self, epoch):
        if self.val_loader is None:
            return None, None

        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0

        # Use torch.inference_mode() for better performance than torch.no_grad()
        with torch.inference_mode():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Fixed autocast for validation
                with amp.autocast(
                    device_type=self.device_type, 
                    dtype=self.autocast_dtype, 
                    enabled=self.mixed_precision
                ):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                total_correct += preds.eq(labels).sum().item()
                total_samples += labels.size(0)

                # Update progress bar
                current_acc = total_correct / total_samples
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.4f}")

        return total_loss / total_samples, total_correct / total_samples

    def fit(self, epochs=10):
        if self.train_loader is None:
            raise ValueError("train_loader must be provided for training.")

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        print(f"\n🚀 Starting training for {epochs} epochs...")
        print("=" * 70)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            # Store history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            if val_loss is not None:
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            # Print epoch results
            if val_loss is not None:
                print(f"Epoch [{epoch:3d}/{epochs}] "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
                      f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch [{epoch:3d}/{epochs}] "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        print("=" * 70)
        print("🎉 Training completed!")
        return history

    def save_model(self, path, epoch=None, **kwargs):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': self.device,
            'mixed_precision': self.mixed_precision,
            **kwargs
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
            
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        epoch = checkpoint.get('epoch', 0)
        print(f"📁 Model loaded from {path} (epoch {epoch})")
        return epoch

    def evaluate(self, test_loader, verbose=True):
        """
        Evaluate the model on a test dataset
        
        Args:
            test_loader: DataLoader for test data
            verbose: Whether to show progress bar and print results
            
        Returns:
            tuple: (test_loss, test_accuracy)
        """
        if test_loader is None:
            raise ValueError("test_loader must be provided for evaluation.")
            
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0

        with torch.inference_mode():
            if verbose:
                pbar = tqdm(test_loader, desc="Evaluating", leave=True)
                iterator = pbar
            else:
                iterator = test_loader
                
            for images, labels in iterator:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Use autocast for consistency with training
                with amp.autocast(
                    device_type=self.device_type, 
                    dtype=self.autocast_dtype, 
                    enabled=self.mixed_precision
                ):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                total_correct += preds.eq(labels).sum().item()
                total_samples += labels.size(0)

                if verbose:
                    current_acc = total_correct / total_samples
                    pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.4f}")

        test_loss = total_loss / total_samples
        test_acc = total_correct / total_samples
        
        if verbose:
            print("=" * 50)
            print(f"📊 Test Results:")
            print(f"   Test Loss: {test_loss:.4f}")
            print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
            print(f"   Correct Predictions: {total_correct}/{total_samples}")
            print("=" * 50)

        return test_loss, test_acc

    def predict(self, data_loader, return_probabilities=False):
        """
        Make predictions on a dataset
        
        Args:
            data_loader: DataLoader for prediction data
            return_probabilities: If True, return softmax probabilities instead of class predictions
            
        Returns:
            numpy.ndarray: Predictions or probabilities
        """
        self.model.eval()
        all_predictions = []
        
        with torch.inference_mode():
            pbar = tqdm(data_loader, desc="Predicting", leave=True)
            for images, _ in pbar:
                images = images.to(self.device, non_blocking=True)
                
                with amp.autocast(
                    device_type=self.device_type, 
                    dtype=self.autocast_dtype, 
                    enabled=self.mixed_precision
                ):
                    outputs = self.model(images)
                
                if return_probabilities:
                    probs = torch.softmax(outputs, dim=1)
                    all_predictions.extend(probs.cpu().numpy())
                else:
                    _, preds = outputs.max(1)
                    all_predictions.extend(preds.cpu().numpy())
        
        import numpy as np
        return np.array(all_predictions)

    def get_model_info(self):
        """Print detailed model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("=" * 60)
        print("🔍 Model Information:")
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Trainable Parameters: {trainable_params:,}")
        print(f"   Non-trainable Parameters: {total_params - trainable_params:,}")
        print(f"   Device: {self.device}")
        print(f"   Mixed Precision: {self.mixed_precision}")
        print(f"   Model Type: {type(self.model).__name__}")
        
        # Calculate model size in MB
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        print(f"   Model Size: {model_size:.2f} MB")
        print("=" * 60)
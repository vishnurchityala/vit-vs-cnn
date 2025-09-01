import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
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
        # Device detection
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Set autocast device_type
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"

        # Verbosity
        print("=" * 60)
        if self.device_type == "cuda":
            print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[INFO] Using CPU (CUDA not available)")
        print("=" * 60)

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training components
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = scheduler

        # Mixed precision
        self.mixed_precision = mixed_precision and self.device_type == "cuda"
        self.scaler = amp.GradScaler(enabled=self.mixed_precision)
        if self.mixed_precision:
            print("[INFO] Mixed precision training enabled.")
        else:
            print("[INFO] Mixed precision training disabled.")

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            with amp.autocast(device_type=self.device_type, dtype=torch.float16, enabled=self.mixed_precision):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler:
                self.scheduler.step()

            # Metrics
            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total_correct += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

            pbar.set_postfix(loss=loss.item(), acc=total_correct / total_samples)

        return total_loss / total_samples, total_correct / total_samples

    def validate(self, epoch):
        if self.val_loader is None:
            return None, None

        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                with amp.autocast(device_type=self.device_type, dtype=torch.float16, enabled=self.mixed_precision):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                total_correct += preds.eq(labels).sum().item()
                total_samples += labels.size(0)

                pbar.set_postfix(loss=loss.item(), acc=total_correct / total_samples)

        return total_loss / total_samples, total_correct / total_samples

    def fit(self, epochs=10):
        if self.train_loader is None:
            raise ValueError("train_loader must be provided for training.")

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            if val_loss is not None:
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            if val_loss is not None:
                print(f"Epoch [{epoch}/{epochs}] "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
                      f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch [{epoch}/{epochs}] "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        return history

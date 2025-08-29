import torch
import torch.nn as nn
import torch.optim as optim


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
        Generic training loop wrapper for PyTorch models with default values.
        
        Args:
            model (nn.Module): The PyTorch model to train.
            train_loader (DataLoader): Training dataloader (default: None).
            val_loader (DataLoader): Validation dataloader (optional, default: None).
            criterion (loss): Loss function (default: CrossEntropyLoss).
            optimizer (torch.optim): Optimizer (default: Adam).
            scheduler: LR scheduler (optional, default: None).
            device (str): 'cuda' or 'cpu' (auto-detect if None).
            mixed_precision (bool): Use AMP for faster training on GPUs (default: False).
            lr (float): Learning rate for default optimizer (default: 3e-4).
            weight_decay (float): Weight decay for default optimizer (default: 0.05).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

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
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    def train_one_epoch(self):
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total_correct += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def validate(self):
        if self.val_loader is None:
            return None, None

        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                total_correct += preds.eq(labels).sum().item()
                total_samples += labels.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def fit(self, epochs=10):
        if self.train_loader is None:
            raise ValueError("train_loader must be provided for training.")

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            if val_loss is not None:
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
                  f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}" if val_loss else
                  f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        return history

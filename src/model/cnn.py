import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ResNetMLP(nn.Module):
    def __init__(self, num_classes, pretrained=True, mlp_hidden=None, dropout=0.2, freeze_backbone=False, device=None):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = []
            
        # Graceful device detection
        self.device = self._detect_device(device)

        # Handle weights (new API instead of deprecated pretrained=True)
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)

        # Replace final FC with identity to get backbone features
        backbone_out = model.fc.in_features
        modules = list(model.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom MLP head
        layers = []
        input_dim = backbone_out
        for h in mlp_hidden:
            layers += [nn.Linear(input_dim, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            input_dim = h
        layers.append(nn.Linear(input_dim, num_classes))
        self.mlp = nn.Sequential(*layers)
        
        # Move model to device with error handling
        try:
            self.backbone = self.backbone.to(self.device)
            self.mlp = self.mlp.to(self.device)
            print(f"[INFO] ResNet model loaded on {self.device}")
        except Exception as e:
            print(f"[WARNING] Failed to move ResNet to {self.device}: {e}, falling back to CPU")
            self.device = "cpu"
            self.backbone = self.backbone.to(self.device)
            self.mlp = self.mlp.to(self.device)
    
    def _detect_device(self, device=None):
        """Gracefully detect device for ResNet"""
        if device is not None:
            if device.startswith("cuda"):
                if not torch.cuda.is_available():
                    print(f"[WARNING] CUDA requested but not available for ResNet, using CPU")
                    return "cpu"
                try:
                    # Test CUDA functionality
                    test_tensor = torch.tensor([1.0]).to(device)
                    return device
                except Exception as e:
                    print(f"[WARNING] CUDA device {device} not accessible for ResNet: {e}, using CPU")
                    return "cpu"
            return device
            
        # Auto-detect: prefer CUDA if available and working
        if torch.cuda.is_available():
            try:
                # Test CUDA functionality
                test_tensor = torch.tensor([1.0]).cuda()
                return "cuda"
            except Exception as e:
                print(f"[WARNING] CUDA available but not functional for ResNet: {e}, using CPU")
                return "cpu"
        else:
            return "cpu"

    def forward(self, x):
        # Safe device handling in forward pass
        try:
            x = x.to(self.device)
            x = self.backbone(x)
            x = torch.flatten(x, 1)
            x = self.mlp(x)
            return x
        except Exception as e:
            print(f"[ERROR] ResNet forward pass failed: {e}")
            # Try CPU fallback if not already on CPU
            if self.device != "cpu":
                print("[INFO] Attempting CPU fallback for ResNet forward pass")
                x = x.to("cpu")
                self.backbone = self.backbone.to("cpu")
                self.mlp = self.mlp.to("cpu")
                self.device = "cpu"
                x = self.backbone(x)
                x = torch.flatten(x, 1)
                x = self.mlp(x)
                return x
            else:
                raise e

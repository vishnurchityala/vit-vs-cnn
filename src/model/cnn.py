import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights


def get_device():
    """Determine the appropriate device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class ResNetMLP(nn.Module):
    """
    ResNet-based classifier with MLP head.
    Clean implementation with proper device handling.
    """
    
    def __init__(self, 
                 num_classes, 
                 pretrained=True, 
                 mlp_hidden=[512, 256], 
                 dropout=0.5, 
                 freeze_backbone=False,
                 device=None):
        super().__init__()
        
        if mlp_hidden is None:
            mlp_hidden = []
        
        # Device handling
        self.device = device if device is not None else get_device()
        
        # Load ResNet50 backbone
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Extract backbone (remove final classifier)
        backbone_out = resnet.fc.in_features
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"[INFO] ResNet backbone frozen")
        
        # Build MLP head
        layers = []
        input_dim = backbone_out
        
        for hidden_dim in mlp_hidden:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        
        # Move to device
        self.to(self.device)
        print(f"[INFO] ResNet model loaded on {self.device}")
    
    def forward(self, x):
        """Forward pass."""
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Extract features
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        
        # Classify
        x = self.classifier(x)
        return x
    
    def get_device(self):
        """Get the device of the model."""
        return self.device

class EfficientNetMLP(nn.Module):
    """
    EfficientNet-B0 backbone with custom MLP head.
    Good for texture-focused datasets.
    """
    def __init__(self,
                 num_classes,
                 pretrained=True,
                 mlp_hidden=[512, 256],
                 dropout=0.5,
                 freeze_backbone=False,
                 device=None):
        super().__init__()

        self.device = device if device is not None else get_device()

        # Load pretrained EfficientNet-B0
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Extract backbone features (remove classifier)
        backbone_out = backbone.classifier[1].in_features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Freeze backbone if needed
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[INFO] EfficientNet backbone frozen")

        # Build MLP head
        layers = []
        input_dim = backbone_out
        if mlp_hidden is None:
            mlp_hidden = []

        for hidden_dim in mlp_hidden:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

        # Move to device
        self.to(self.device)
        print(f"[INFO] EfficientNet-B0 model loaded on {self.device}")

    def forward(self, x):
        # Ensure input is on correct device
        x = x.to(self.device)

        # Extract features
        x = self.backbone(x)
        x = torch.flatten(x, 1)

        # Classify
        x = self.classifier(x)
        return x

    def get_device(self):
        """Get the device of the model."""
        return self.device
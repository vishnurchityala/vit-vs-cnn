import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ResNetMLP(nn.Module):
    def __init__(self, num_classes, pretrained=True, mlp_hidden=None, dropout=0.2, freeze_backbone=False):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = []

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

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x

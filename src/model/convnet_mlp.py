import torch
import torch.nn as nn
import timm

class ConvNeXtXXL_MLP(nn.Module):
    def __init__(self, num_classes=47, pretrained=True, mlp_hidden=None, dropout=0.5, freeze_backbone=True):
        super().__init__()

        if mlp_hidden is None:
            mlp_hidden = [1024,512,256]

        # pretrained ConvNeXt-XXL backbone
        self.backbone = timm.create_model("convnext_xxlarge", pretrained=pretrained)

        in_features = self.backbone.head.fc.in_features

        self.backbone.head.fc = nn.Identity()

        layers = []
        input_dim = in_features
        for h in mlp_hidden:
            layers += [
                nn.Linear(input_dim, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ]
            input_dim = h
        layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

        if freeze_backbone:
            for param in self.backbone.stem.parameters():
                param.requires_grad = False
            for param in self.backbone.stages.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)     # Extract features
        x = self.classifier(x)   # Pass through custom MLP head
        return x
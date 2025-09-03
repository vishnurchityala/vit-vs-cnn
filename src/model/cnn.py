import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetMLP(nn.Module):
    def __init__(self, 
                 num_classes, 
                 pretrained=True, 
                 mlp_hidden=[256], 
                 dropout=0.7, 
                 freeze_backbone=True, 
                 unfreeze_layers=1,
                 label_smoothing=0.2,
                 device=None):
        super().__init__()
        self.device = device if device is not None else get_device()
        self.label_smoothing = label_smoothing


        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        backbone_out = resnet.fc.in_features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])


        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False


            resnet_layers = list(self.backbone.children())
            if unfreeze_layers > 0:
                for layer in resnet_layers[-unfreeze_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True


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

        self.to(self.device)


        self.optimizer = optim.AdamW([
            {"params": self.backbone.parameters(), "lr": 1e-5},
            {"params": self.classifier.parameters(), "lr": 1e-4}
        ], weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)

    def forward(self, x):
        x = x.to(self.device)
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def step_scheduler(self, val_loss):
        self.scheduler.step(val_loss)

    def compute_loss(self, outputs, labels):

        if self.label_smoothing > 0:
            n_classes = outputs.size(1)
            smooth_labels = torch.full_like(outputs, self.label_smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - self.label_smoothing)
            log_probs = F.log_softmax(outputs, dim=1)
            loss = -(smooth_labels * log_probs).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(outputs, labels)
        

        confidence_penalty = 0.1 * (outputs.max(dim=1)[0]).mean()
        loss += confidence_penalty
        
        return loss

    def get_device(self):
        return self.device


class EfficientNetMLP(nn.Module):
    def __init__(self,
                 num_classes,
                 pretrained=True,
                 mlp_hidden=[256],
                 dropout=0.7,
                 freeze_backbone=True,
                 unfreeze_layers=0,  # Keep frozen for better regularization
                 label_smoothing=0.2,
                 device=None):
        super().__init__()
        self.device = device if device is not None else get_device()
        self.label_smoothing = label_smoothing

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        backbone_out = backbone.classifier[1].in_features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])


        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False


            if unfreeze_layers > 0:

                if hasattr(backbone, 'features'):
                    for block in self.backbone[-1].children()[-unfreeze_layers:]:
                        for param in block.parameters():
                            param.requires_grad = True


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

        self.to(self.device)


        self.optimizer = optim.AdamW([
            {"params": self.backbone.parameters(), "lr": 1e-5},
            {"params": self.classifier.parameters(), "lr": 1e-4}
        ], weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)

    def forward(self, x):
        x = x.to(self.device)
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def step_scheduler(self, val_loss):
        self.scheduler.step(val_loss)

    def compute_loss(self, outputs, labels):

        if self.label_smoothing > 0:
            n_classes = outputs.size(1)
            smooth_labels = torch.full_like(outputs, self.label_smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - self.label_smoothing)
            log_probs = F.log_softmax(outputs, dim=1)
            loss = -(smooth_labels * log_probs).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(outputs, labels)
        

        confidence_penalty = 0.1 * (outputs.max(dim=1)[0]).mean()
        loss += confidence_penalty
        
        return loss

    def get_device(self):
        return self.device

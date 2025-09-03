import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision.models import ViT_B_16_Weights, ViT_L_16_Weights

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resize_positional_embedding(old_embedding, new_size, hidden_dim):
    num_extra_tokens = 1
    old_num_patches = old_embedding.shape[1] - num_extra_tokens
    old_size = int(old_num_patches ** 0.5)

    cls_token = old_embedding[:, :num_extra_tokens, :]
    patch_tokens = old_embedding[:, num_extra_tokens:, :]
    patch_tokens = patch_tokens.reshape(1, old_size, old_size, hidden_dim).permute(0, 3, 1, 2)

    new_patch_tokens = F.interpolate(
        patch_tokens,
        size=(new_size, new_size),
        mode="bicubic",
        align_corners=False
    )

    new_patch_tokens = new_patch_tokens.permute(0, 2, 3, 1).reshape(1, new_size * new_size, hidden_dim)
    return nn.Parameter(torch.cat([cls_token, new_patch_tokens], dim=1))

class CustomViTClassifier(nn.Module):
    def __init__(self,
                 num_classes,
                 model_name="vit_b_16",
                 pretrained=True,
                 freeze_backbone=True,
                 unfreeze_layers=3,
                 img_size=224,
                 dropout=0.7,
                 label_smoothing=0.2,
                 device=None):
        super().__init__()
        self.device = device if device is not None else get_device()
        self.img_size = img_size
        self.model_name = model_name
        self.label_smoothing = label_smoothing

        if model_name == "vit_b_16":
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
        elif model_name == "vit_l_16":
            weights = ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vit_l_16(weights=weights)
        else:
            raise ValueError("Unsupported model_name. Use 'vit_b_16' or 'vit_l_16'.")

        self.hidden_dim = self.backbone.hidden_dim
        self.patch_size = self.backbone.patch_size
        num_patches = (img_size // self.patch_size) ** 2

        if pretrained and img_size != 224:
            self.backbone.encoder.pos_embedding = resize_positional_embedding(
                self.backbone.encoder.pos_embedding.data,
                img_size // self.patch_size,
                self.hidden_dim
            )
        elif not pretrained:
            self.backbone.encoder.pos_embedding = nn.Parameter(
                torch.zeros(1, num_patches + 1, self.hidden_dim)
            )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if unfreeze_layers > 0:
                total_blocks = len(self.backbone.encoder.layers)
                for i in range(total_blocks - unfreeze_layers, total_blocks):
                    for param in self.backbone.encoder.layers[i].parameters():
                        param.requires_grad = True

        self.backbone.heads = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, num_classes)
        )

        self.to(self.device)

        self.optimizer = optim.AdamW([
            {"params": self.backbone.parameters(), "lr": 1e-5},
            {"params": self.classifier.parameters(), "lr": 1e-4}
        ], weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    def forward(self, x):
        x = x.to(self.device)
        features = self.backbone(x)
        output = self.classifier(features)
        return output

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

    def step_scheduler(self, val_loss):
        self.scheduler.step(val_loss)

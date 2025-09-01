import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def resize_positional_embedding(old_emb, new_size, hidden_dim):
    """
    Resize pretrained positional embeddings to new image size using bicubic interpolation.
    """
    num_extra_tokens = 1  # class token
    old_num_patches = old_emb.shape[1] - num_extra_tokens
    old_size = int(old_num_patches ** 0.5)

    # Separate cls token and patch tokens
    cls_token = old_emb[:, :num_extra_tokens]
    patch_tokens = old_emb[:, num_extra_tokens:]

    # Reshape patch tokens -> (1, hidden_dim, H, W)
    patch_tokens = patch_tokens.reshape(1, old_size, old_size, hidden_dim).permute(0, 3, 1, 2)

    # Interpolate
    new_patch_tokens = F.interpolate(
        patch_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
    )

    # Reshape back
    new_patch_tokens = new_patch_tokens.permute(0, 2, 3, 1).reshape(1, new_size * new_size, hidden_dim)

    # Concatenate cls token
    return nn.Parameter(torch.cat([cls_token, new_patch_tokens], dim=1))

class CustomViTClassifier(nn.Module):
    """
    Unified ViT classifier with flexible output head and positional embedding resizing.
    Uses torchvision.models VisionTransformer backbone.
    """
    def __init__(
        self,
        num_classes,
        model_name="vit_b_16",
        pretrained=True,
        freeze_backbone=True,
        img_size=224,
        device=None
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.img_size = img_size

        # Load model with or without pretrained weights
        if model_name == "vit_b_16":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.vit_b_16(weights=weights)
        elif model_name == "vit_l_16":
            weights = models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.vit_l_16(weights=weights)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace head with identity for feature extraction
        self.model.heads = nn.Identity()
        patch_size = self.model.patch_size
        hidden_dim = self.model.hidden_dim
        num_patches = (img_size // patch_size) * (img_size // patch_size)

        # Resize positional embedding if needed (for pretrained case)
        if pretrained and img_size != 224:
            self.model.encoder.pos_embedding = resize_positional_embedding(
                self.model.encoder.pos_embedding.data, img_size // patch_size, hidden_dim
            )
        elif not pretrained:
            self.model.encoder.pos_embedding = nn.Parameter(
                torch.zeros(1, num_patches + 1, hidden_dim)
            )

        # Custom classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.model = self.model.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        features = self.model(x)
        return self.classifier(features)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def resize_positional_embedding(old_emb, new_size, patch_size, hidden_dim):
    """
    Resize pretrained positional embeddings to new image size using interpolation.
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


# ------------------------------
# Pretrained Feature Extractor
# ------------------------------
class PreTrainedViTFeatureExtractor(nn.Module):
    """
    PreTrainedViTFeatureExtractor with customizable input size.
    Keeps pretrained positional embeddings (resized if necessary).
    """
    def __init__(self, model_name="vit_b_16", pretrained=True, freeze=True, img_size=224, device=None):
        super(PreTrainedViTFeatureExtractor, self).__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size

        # Load model
        if pretrained:
            if model_name == "vit_b_16":
                weights = models.ViT_B_16_Weights.IMAGENET1K_V1
                self.model = models.vit_b_16(weights=weights)
            elif model_name == "vit_l_16":
                weights = models.ViT_L_16_Weights.IMAGENET1K_V1
                self.model = models.vit_l_16(weights=weights)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        else:
            if model_name == "vit_b_16":
                self.model = models.vit_b_16(weights=None)
            elif model_name == "vit_l_16":
                self.model = models.vit_l_16(weights=None)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

        # Freeze backbone if needed
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace classifier head with identity
        self.model.heads = nn.Identity()

        # Handle positional embeddings
        patch_size = self.model.patch_size
        hidden_dim = self.model.encoder.pos_embedding.shape[-1]
        new_size = img_size // patch_size

        if pretrained and img_size != 224:  # interpolate pretrained embeddings
            self.model.encoder.pos_embedding = resize_positional_embedding(
                self.model.encoder.pos_embedding.data, new_size, patch_size, hidden_dim
            )
        elif not pretrained:  # random reinit
            num_patches = (img_size // patch_size) * (img_size // patch_size)
            self.model.encoder.pos_embedding = nn.Parameter(
                torch.zeros(1, num_patches + 1, hidden_dim)
            )

        self.model = self.model.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)


class PreTrainedViTClassifier(nn.Module):
    def __init__(self, num_classes, model_name="vit_b_16", img_size=224, device=None):
        super(PreTrainedViTClassifier, self).__init__()
        self.backbone = PreTrainedViTFeatureExtractor(
            model_name=model_name, img_size=img_size, freeze=True, device=device
        )
        hidden_dim = self.backbone.model.hidden_dim
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ------------------------------
# Custom (Untrained) Feature Extractor
# ------------------------------
class CustomViTFeatureExtractor(nn.Module):
    """
    Custom (Untrained) Vision Transformer Feature Extractor.
    Randomly initialized, supports custom input size.
    """
    def __init__(self, model_name="vit_b_16", img_size=224, freeze=False, device=None):
        super(CustomViTFeatureExtractor, self).__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size

        # Load untrained model
        if model_name == "vit_b_16":
            self.model = models.vit_b_16(weights=None)
        elif model_name == "vit_l_16":
            self.model = models.vit_l_16(weights=None)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Optionally freeze
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace head with identity
        self.model.heads = nn.Identity()

        # Reinit positional embeddings
        patch_size = self.model.patch_size
        hidden_dim = self.model.encoder.pos_embedding.shape[-1]
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.model.encoder.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, hidden_dim)
        )

        self.model = self.model.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)


class CustomViTClassifier(nn.Module):
    def __init__(self, num_classes, model_name="vit_b_16", img_size=224, device=None):
        super(CustomViTClassifier, self).__init__()
        self.backbone = CustomViTFeatureExtractor(
            model_name=model_name, img_size=img_size, freeze=False, device=device
        )
        hidden_dim = self.backbone.model.hidden_dim
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


import torch

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dummy input: batch of 4 RGB images of size 224x224
    x = torch.randn(4, 3, 224, 224).to(device)

    print("=== Pretrained Feature Extractor ===")
    pretrained_feat = PreTrainedViTFeatureExtractor(
        model_name="vit_b_16", pretrained=True, freeze=True, img_size=224, device=device
    )
    features = pretrained_feat(x)
    print("Feature shape:", features.shape)  # e.g. (4, 768)

    print("\n=== Pretrained Classifier ===")
    pretrained_cls = PreTrainedViTClassifier(
        num_classes=10, model_name="vit_b_16", img_size=224, device=device
    )
    logits = pretrained_cls(x)
    print("Logits shape:", logits.shape)  # (4, 10)

    print("\n=== Custom (Untrained) Feature Extractor ===")
    custom_feat = CustomViTFeatureExtractor(
        model_name="vit_b_16", img_size=224, freeze=False, device=device
    )
    features = custom_feat(x)
    print("Feature shape:", features.shape)  # (4, 768)

    print("\n=== Custom (Untrained) Classifier ===")
    custom_cls = CustomViTClassifier(
        num_classes=10, model_name="vit_b_16", img_size=224, device=device
    )
    logits = custom_cls(x)
    print("Logits shape:", logits.shape)  # (4, 10)

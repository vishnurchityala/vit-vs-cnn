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
    Unified ViT classifier with graceful device handling.
    Platform-agnostic for Mac development and Windows server training.
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
        # Graceful device detection and validation
        self.device = self._detect_and_validate_device(device)
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
        
        # Safely move model to device with error handling
        try:
            self.model = self.model.to(self.device, dtype=torch.float32)
            self.classifier = self.classifier.to(self.device, dtype=torch.float32)
            print(f"[INFO] ViT model loaded on {self.device}")
        except Exception as e:
            print(f"[WARNING] Failed to move ViT to {self.device}: {e}, falling back to CPU")
            self.device = "cpu"
            self.model = self.model.to(self.device, dtype=torch.float32)
            self.classifier = self.classifier.to(self.device, dtype=torch.float32)

    def _detect_and_validate_device(self, device=None):
        """Gracefully detect and validate device for ViT"""
        if device is not None:
            if device.startswith("cuda"):
                if not torch.cuda.is_available():
                    print(f"[WARNING] CUDA requested but not available for ViT, using CPU")
                    return "cpu"
                try:
                    # Test CUDA functionality
                    test_tensor = torch.tensor([1.0]).to(device)
                    return device
                except Exception as e:
                    print(f"[WARNING] CUDA device {device} not accessible for ViT: {e}, using CPU")
                    return "cpu"
            return device
            
        # Auto-detect: prefer CUDA if available and working
        if torch.cuda.is_available():
            try:
                # Test CUDA functionality
                test_tensor = torch.tensor([1.0]).cuda()
                return "cuda"
            except Exception as e:
                print(f"[WARNING] CUDA available but not functional for ViT: {e}, using CPU")
                return "cpu"
        else:
            return "cpu"

    def forward(self, x):
        # Safe device and dtype handling
        try:
            x = x.to(self.device, dtype=torch.float32)
            features = self.model(x)
            return self.classifier(features)
        except Exception as e:
            print(f"[ERROR] Forward pass failed: {e}")
            # Try CPU fallback if not already on CPU
            if self.device != "cpu":
                print("[INFO] Attempting CPU fallback for ViT forward pass")
                x = x.to("cpu", dtype=torch.float32)
                self.model = self.model.to("cpu")
                self.classifier = self.classifier.to("cpu")
                self.device = "cpu"
                features = self.model(x)
                return self.classifier(features)
            else:
                raise e

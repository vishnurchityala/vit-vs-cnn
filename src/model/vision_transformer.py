import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights, ViT_L_16_Weights


def get_device():
    """Determine the appropriate device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def resize_positional_embedding(old_embedding, new_size, hidden_dim):
    """
    Resize pretrained positional embeddings to new image size using bicubic interpolation.
    
    Args:
        old_embedding: Original positional embedding tensor
        new_size: New spatial size (height and width)
        hidden_dim: Hidden dimension of the embeddings
    
    Returns:
        Resized positional embedding as nn.Parameter
    """
    num_extra_tokens = 1  # CLS token
    old_num_patches = old_embedding.shape[1] - num_extra_tokens
    old_size = int(old_num_patches ** 0.5)
    
    # Separate CLS token and patch tokens
    cls_token = old_embedding[:, :num_extra_tokens, :]
    patch_tokens = old_embedding[:, num_extra_tokens:, :]
    
    # Reshape patch tokens to 2D grid
    patch_tokens = patch_tokens.reshape(1, old_size, old_size, hidden_dim).permute(0, 3, 1, 2)
    
    # Interpolate to new size
    new_patch_tokens = F.interpolate(
        patch_tokens, 
        size=(new_size, new_size), 
        mode="bicubic", 
        align_corners=False
    )
    
    # Reshape back to sequence
    new_patch_tokens = new_patch_tokens.permute(0, 2, 3, 1).reshape(1, new_size * new_size, hidden_dim)
    
    # Concatenate CLS token and new patch tokens
    return nn.Parameter(torch.cat([cls_token, new_patch_tokens], dim=1))


class CustomViTClassifier(nn.Module):
    """
    Vision Transformer classifier with flexible output head.
    Clean implementation with proper device handling.
    """
    
    def __init__(self,
                 num_classes,
                 model_name="vit_b_16",
                 pretrained=True,
                 freeze_backbone=True,
                 img_size=224,
                 device=None):
        super().__init__()
        
        # Device handling
        self.device = device if device is not None else get_device()
        self.model_name = model_name
        self.img_size = img_size
        
        # Load pretrained ViT model
        if model_name == "vit_b_16":
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
        elif model_name == "vit_l_16":
            weights = ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vit_l_16(weights=weights)
        else:
            raise ValueError(f"Unsupported model: {model_name}. Use 'vit_b_16' or 'vit_l_16'")
        
        # Get model properties
        self.patch_size = self.backbone.patch_size
        self.hidden_dim = self.backbone.hidden_dim
        num_patches = (img_size // self.patch_size) ** 2
        
        # Handle positional embedding resizing if needed
        if pretrained and img_size != 224:
            print(f"[INFO] Resizing positional embeddings from 224x224 to {img_size}x{img_size}")
            self.backbone.encoder.pos_embedding = resize_positional_embedding(
                self.backbone.encoder.pos_embedding.data, 
                img_size // self.patch_size, 
                self.hidden_dim
            )
        elif not pretrained:
            # Initialize new positional embeddings
            self.backbone.encoder.pos_embedding = nn.Parameter(
                torch.zeros(1, num_patches + 1, self.hidden_dim)
            )
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"[INFO] ViT backbone frozen")
        
        # Replace the original head with identity for feature extraction
        self.backbone.heads = nn.Identity()
        
        # Custom classifier head
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        
        # Move to device
        self.to(self.device)
        print(f"[INFO] ViT model loaded on {self.device}")
    
    def forward(self, x):
        """Forward pass."""
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Extract features using ViT backbone
        features = self.backbone(x)  # Shape: [batch_size, hidden_dim]
        
        # Classify
        output = self.classifier(features)
        return output
    
    def get_device(self):
        """Get the device of the model."""
        return self.device
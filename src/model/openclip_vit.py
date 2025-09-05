import torch
import torch.nn as nn
import open_clip


class OpenCLIPViTL14_MLP(nn.Module):
    def __init__(self, num_classes=47, pretrained=True, mlp_hidden=None, dropout=0.5, freeze_backbone=True):
        super().__init__()

        # Default hidden layers if none provided
        if mlp_hidden is None:
            mlp_hidden = [1024, 512]

        # Load pretrained OpenCLIP ViT-L/14 backbone with OpenAI pretraining
        if pretrained:
            self.backbone, _, _ = open_clip.create_model_and_transforms(
                'ViT-L-14', 
                pretrained='openai'
            )
        else:
            self.backbone, _, _ = open_clip.create_model_and_transforms('ViT-L-14')

        # Extract feature dimension from the vision transformer
        # ViT-L/14 has an embedding dimension of 1024
        in_features = self.backbone.visual.output_dim
        
        # Remove the original text projection head as we only need vision features
        # Keep only the visual encoder
        self.visual_encoder = self.backbone.visual

        # Build custom MLP classifier
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

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Extract visual features using the OpenCLIP vision transformer
        x = self.visual_encoder(x)  # Extract features
        x = self.classifier(x)      # Pass through custom MLP head
        return x

    def unfreeze_layers(self, num_layers=1):
        """
        Unfreeze the last few transformer blocks for fine-tuning.
        
        Args:
            num_layers (int): Number of transformer blocks to unfreeze from the end
        """
        # Unfreeze the last num_layers transformer blocks
        transformer_blocks = list(self.visual_encoder.transformer.resblocks)
        for block in transformer_blocks[-num_layers:]:
            for param in block.parameters():
                param.requires_grad = True
        
        # Also unfreeze the final layer norm and projection if they exist
        if hasattr(self.visual_encoder, 'ln_post'):
            for param in self.visual_encoder.ln_post.parameters():
                param.requires_grad = True
                
        if hasattr(self.visual_encoder, 'proj') and self.visual_encoder.proj is not None:
            for param in self.visual_encoder.proj.parameters():
                param.requires_grad = True

    def get_feature_extractor(self):
        """
        Return just the visual encoder for feature extraction.
        """
        return self.visual_encoder

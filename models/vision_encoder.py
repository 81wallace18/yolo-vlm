import torch
import torch.nn as nn
from transformers import CLIPVisionModel


class FrozenVisionEncoder(nn.Module):
    """CLIP ViT-B/32 vision encoder with all weights frozen."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Returns pooled CLS token: (B, vision_dim)
        outputs = self.model(pixel_values=pixel_values)
        return outputs.pooler_output

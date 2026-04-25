import torch.nn as nn


class ProjectionMLP(nn.Module):
    """Projects vision features into the language model embedding space."""

    def __init__(self, vision_dim: int, hidden_dim: int, language_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, language_dim),
        )

    def forward(self, x):
        # x: (B, vision_dim) → (B, 1, language_dim)
        return self.net(x).unsqueeze(1)

import torch
import torch.nn as nn

from models.vision_encoder import FrozenVisionEncoder
from models.projection import ProjectionMLP
from models.language_decoder import LoRALanguageDecoder


class SmallVLM(nn.Module):
    """Small Vision-Language Model.

    Pipeline:
        pixel_values → FrozenVisionEncoder → ProjectionMLP
                                                     ↓
        input_ids → token embeddings → concat([vision_token, text_tokens]) → LoRADecoder → loss / logits
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.vision_encoder = FrozenVisionEncoder(m["vision_encoder"])
        self.projection = ProjectionMLP(
            vision_dim=m["vision_dim"],
            hidden_dim=m["projection_hidden_dim"],
            language_dim=m["language_dim"],
        )
        self.decoder = LoRALanguageDecoder(
            model_name=m["language_model"],
            lora_rank=m["lora_rank"],
            lora_alpha=m.get("lora_alpha", 16),
            lora_dropout=m.get("lora_dropout", 0.05),
        )
        self.max_new_tokens = m.get("max_new_tokens", 64)

    def _lm_dtype(self):
        return next(self.decoder.parameters()).dtype

    def _build_inputs_embeds(self, pixel_values, input_ids):
        """Prepend a single vision token to the text token embeddings."""
        dtype = self._lm_dtype()
        vision_features = self.vision_encoder(pixel_values)             # (B, vision_dim) float32
        vision_token = self.projection(vision_features).to(dtype)       # (B, 1, lang_dim) → lm dtype
        token_embeds = self.decoder.get_input_embeddings()(input_ids)   # (B, T, lang_dim)
        return torch.cat([vision_token, token_embeds], dim=1)           # (B, 1+T, lang_dim)

    def forward(self, pixel_values, input_ids, attention_mask=None):
        inputs_embeds = self._build_inputs_embeds(pixel_values, input_ids)

        # Mask padding in labels using the original attention_mask (B, T) — must happen
        # before extending it, so shapes match for boolean indexing.
        labels = input_ids.clone()
        if attention_mask is not None:
            labels[attention_mask == 0] = -100

        # Extend attention mask to cover the prepended vision token → (B, 1+T)
        if attention_mask is not None:
            B = input_ids.size(0)
            vision_mask = torch.ones(B, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # Prepend -100 for the vision token position so loss ignores it → (B, 1+T)
        B = input_ids.size(0)
        vision_label = torch.full((B, 1), -100, device=labels.device, dtype=labels.dtype)
        labels = torch.cat([vision_label, labels], dim=1)

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, pixel_values, input_ids=None, attention_mask=None):
        """Generate a caption for the given image."""
        dtype = self._lm_dtype()
        vision_features = self.vision_encoder(pixel_values)
        vision_token = self.projection(vision_features).to(dtype)  # (B, 1, lang_dim)

        if input_ids is not None:
            token_embeds = self.decoder.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([vision_token, token_embeds], dim=1)
            if attention_mask is not None:
                B = input_ids.size(0)
                vision_mask = torch.ones(B, 1, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        else:
            inputs_embeds = vision_token
            B = vision_token.shape[0]
            if attention_mask is None:
                attention_mask = torch.ones(B, inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long)

        return self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
        )

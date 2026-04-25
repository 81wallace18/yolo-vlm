import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


class LoRALanguageDecoder(nn.Module):
    """Small causal LM with LoRA adapters for efficient fine-tuning."""

    def __init__(self, model_name: str, lora_rank: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05):
        super().__init__()
        base = AutoModelForCausalLM.from_pretrained(model_name)
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.model = get_peft_model(base, lora_cfg)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    def generate(self, inputs_embeds, attention_mask=None, max_new_tokens: int = 64, **kwargs):
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **kwargs,
        )

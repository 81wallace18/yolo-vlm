import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


class Phi3VLM(nn.Module):
    """Fine-tunable wrapper for microsoft/Phi-3-vision-128k-instruct.

    Phi-3-vision is a complete VLM — it handles image encoding internally.
    We apply LoRA on the attention layers and fine-tune on YOLO captions.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        model_name = m["language_model"]

        bnb_config = None
        if m.get("load_in_4bit", False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            _attn_implementation="eager",
        )

        lora_cfg = LoraConfig(
            r=m["lora_rank"],
            lora_alpha=m.get("lora_alpha", 16),
            lora_dropout=m.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["qkv_proj", "o_proj"],
        )
        self.model = get_peft_model(base, lora_cfg)
        self.max_new_tokens = m.get("max_new_tokens", 64)

    def forward(self, input_ids, attention_mask, pixel_values, image_sizes, labels):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, pixel_values, image_sizes):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            eos_token_id=self.model.config.eos_token_id,
        )

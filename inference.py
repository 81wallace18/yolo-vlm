import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


def _latest_checkpoint(ckpt_dir: str = "checkpoints") -> Path:
    ckpts = sorted(Path(ckpt_dir).glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in '{ckpt_dir}'. Run train.py first.")
    return ckpts[-1]


def parse_args():
    p = argparse.ArgumentParser(description="Run VLM inference on an image")
    p.add_argument("image", help="Path to input image")
    p.add_argument("--checkpoint", default=None, help="Checkpoint .pt (default: latest in checkpoints/)")
    return p.parse_args()


def main():
    args = parse_args()

    ckpt_path = args.checkpoint or _latest_checkpoint()
    print(f"Checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model_type = cfg["model"].get("type", "custom")

    image = Image.open(args.image).convert("RGB")

    if model_type == "phi3":
        from models.phi3_vlm import Phi3VLM

        processor = AutoProcessor.from_pretrained(
            cfg["model"]["language_model"], trust_remote_code=True
        )
        model = Phi3VLM(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        from data.phi3_dataset import _PROMPT
        enc = processor(
            text=_PROMPT,
            images=image,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        pixel_values = enc["pixel_values"].to(device)
        image_sizes = enc.get("image_sizes", torch.tensor([[image.height, image.width]])).to(device)

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )
        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[1]:]
        caption = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    else:
        from models.small_vlm import SmallVLM

        processor = AutoProcessor.from_pretrained(cfg["model"]["vision_encoder"])
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["language_model"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = SmallVLM(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        output_ids = model.generate(pixel_values)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"Image  : {args.image}")
    print(f"Caption: {caption}")


if __name__ == "__main__":
    main()

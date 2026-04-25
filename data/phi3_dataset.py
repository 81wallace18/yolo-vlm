from pathlib import Path
from typing import Union

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from data.label_to_caption import yolo_labels_to_caption

# Phi-3-vision chat template
_PROMPT = "<|user|>\n<|image_1|>\nDescribe the objects in this image.<|end|>\n<|assistant|>\n"
_SUFFIX = "<|end|>"


class Phi3Dataset(Dataset):
    """YOLO-format dataset formatted for Phi-3-vision fine-tuning.

    Returns: {input_ids, attention_mask, pixel_values, image_sizes, labels}
    Labels mask the prompt portion with -100 so loss only covers the caption.
    """

    def __init__(self, yaml_path: Union[str, Path], split: str = "train", processor=None, max_length: int = 256):
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        dataset_root = Path(cfg["path"])
        if not dataset_root.is_absolute():
            dataset_root = Path(yaml_path).parent / dataset_root

        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split

        names = cfg.get("names", {})
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        self.class_names = {int(k): v for k, v in names.items()}

        self.processor = processor
        self.max_length = max_length

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.samples = [
            (p, labels_dir / (p.stem + ".txt"))
            for p in sorted(images_dir.iterdir())
            if p.suffix.lower() in exts
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        caption = yolo_labels_to_caption(str(label_path), self.class_names)

        full_text = _PROMPT + caption + _SUFFIX

        # Tokenize full text + process image
        enc = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        pixel_values = enc["pixel_values"].squeeze(0)
        image_sizes = enc.get("image_sizes", torch.tensor([[image.height, image.width]])).squeeze(0)

        # Mask everything before (and including) <|assistant|>\n so loss covers only the caption.
        # We search in the actual input_ids (which include image tokens) rather than tokenizing
        # the prompt text separately — avoids the off-by-N error from image token insertion.
        labels = input_ids.clone()
        assistant_id = self.processor.tokenizer.convert_tokens_to_ids("<|assistant|>")
        positions = (input_ids == assistant_id).nonzero(as_tuple=True)[0]
        response_start = int(positions[-1]) + 2 if len(positions) > 0 else 0
        labels[:response_start] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "labels": labels,
        }

from pathlib import Path
from typing import Union

import yaml
from PIL import Image
from torch.utils.data import Dataset

from data.label_to_caption import yolo_labels_to_caption


class YOLOVLMDataset(Dataset):
    """Reads a YOLO-format dataset and returns (PIL image, caption string) pairs.

    Dataset layout expected:
        <root>/
            images/train/  or  images/val/
            labels/train/  or  labels/val/

    The yaml file must have keys: path, train, val, names.
    """

    def __init__(self, yaml_path: Union[str, Path], split: str = "train", processor=None, tokenizer=None, max_length: int = 64):
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
        self.tokenizer = tokenizer
        self.max_length = max_length

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.samples = []
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in exts:
                continue
            label_path = labels_dir / (img_path.stem + ".txt")
            self.samples.append((img_path, label_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        caption = yolo_labels_to_caption(str(label_path), self.class_names)

        if self.processor is not None and self.tokenizer is not None:
            pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
            encoding = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}

        return image, caption

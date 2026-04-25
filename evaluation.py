"""Caption-level evaluation metrics for the VLM.

This is NOT mAP in the literal object-detection sense (mAP requires predicted
bboxes + confidence scores + IoU, which a text-generating VLM cannot produce).
What we compute is an honest adaptation:

    - Per-class precision / recall / F1 over the val set, where TP/FP/FN are
      decided by class name match (image-level: did the model mention class C?).
    - macro_F1: average F1 across classes — the closest single-number proxy
      for mAP that makes sense for free-form caption output.
    - position_accuracy: of the classes correctly identified, in how many cases
      did the model also place them in the right grid cell?

Generation is sequential (one image at a time) since model.generate() doesn't
batch cleanly across the two backends; for typical val set sizes this is fine.
"""

from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from data.caption_parser import parse_caption
from data.label_to_caption import yolo_labels_to_caption


def _generate_caption_custom(model, image, processor, tokenizer, device) -> str:
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    output_ids = model.generate(pixel_values=pixel_values)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def _generate_caption_phi3(model, image, processor, device) -> str:
    from data.phi3_dataset import _PROMPT
    enc = processor(text=_PROMPT, images=image, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    pixel_values = enc["pixel_values"].to(device)
    image_sizes = enc.get(
        "image_sizes", torch.tensor([[image.height, image.width]])
    ).to(device)
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
    )
    new_tokens = output_ids[0][input_ids.shape[1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)


def _aggregate(predictions, ground_truths) -> dict:
    """Convert lists of parsed (class, position) tuples into per-class metrics."""
    tp: dict = defaultdict(int)
    fp: dict = defaultdict(int)
    fn: dict = defaultdict(int)
    pos_correct = 0
    pos_total = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_classes = {c for c, _ in pred}
        gt_classes = {c for c, _ in gt}

        for c in pred_classes & gt_classes:
            tp[c] += 1
        for c in pred_classes - gt_classes:
            fp[c] += 1
        for c in gt_classes - pred_classes:
            fn[c] += 1

        pred_pos: dict = defaultdict(set)
        for c, p in pred:
            pred_pos[c].add(p)
        gt_pos: dict = defaultdict(set)
        for c, p in gt:
            gt_pos[c].add(p)

        # Position accuracy: for each class present in both pred and gt,
        # count it correct if any predicted position matches any gt position.
        for c in pred_classes & gt_classes:
            pos_total += 1
            if pred_pos[c] & gt_pos[c]:
                pos_correct += 1

    classes_with_gt: set = set()
    for gt in ground_truths:
        classes_with_gt.update(c for c, _ in gt)

    per_class = {}
    for c in classes_with_gt:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[c] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": tp[c] + fn[c],
        }

    n = max(len(per_class), 1)
    return {
        "per_class": per_class,
        "macro_precision": sum(m["precision"] for m in per_class.values()) / n,
        "macro_recall": sum(m["recall"] for m in per_class.values()) / n,
        "macro_f1": sum(m["f1"] for m in per_class.values()) / n,
        "position_accuracy": pos_correct / pos_total if pos_total > 0 else 0.0,
    }


@torch.no_grad()
def evaluate_captions(model, val_ds, processor, tokenizer, device, model_type: str) -> dict:
    """Generate captions for the val set and compute caption-level metrics."""
    model.eval()
    valid_classes = frozenset(name.lower() for name in val_ds.class_names.values())

    predictions: list = []
    ground_truths: list = []

    for img_path, label_path in val_ds.samples:
        image = Image.open(img_path).convert("RGB")

        if model_type == "phi3":
            generated = _generate_caption_phi3(model, image, processor, device)
        else:
            generated = _generate_caption_custom(model, image, processor, tokenizer, device)

        gt_caption = yolo_labels_to_caption(str(label_path), val_ds.class_names)

        predictions.append(parse_caption(generated, valid_classes))
        ground_truths.append(parse_caption(gt_caption, valid_classes))

    return _aggregate(predictions, ground_truths)


def format_metrics(metrics: dict) -> str:
    """Render metrics as a multi-line string for printing."""
    lines = [
        f"  macro_precision={metrics['macro_precision']:.3f}  "
        f"macro_recall={metrics['macro_recall']:.3f}  "
        f"macro_f1={metrics['macro_f1']:.3f}  "
        f"position_acc={metrics['position_accuracy']:.3f}",
    ]
    if metrics["per_class"]:
        lines.append("  per-class:")
        for c in sorted(metrics["per_class"].keys()):
            m = metrics["per_class"][c]
            lines.append(
                f"    {c:20s}  P={m['precision']:.2f}  R={m['recall']:.2f}  "
                f"F1={m['f1']:.2f}  (n={m['support']})"
            )
    return "\n".join(lines)

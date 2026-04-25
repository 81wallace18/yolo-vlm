import argparse
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, get_cosine_schedule_with_warmup

from data.download import ensure_dataset
from evaluation import evaluate_captions, format_metrics


def build_components(cfg: dict, data_yaml):
    """Factory — returns (model, train_ds, val_ds, processor, tokenizer) for any model.type."""
    model_type = cfg["model"].get("type", "custom")

    if model_type == "phi3":
        from models.phi3_vlm import Phi3VLM
        from data.phi3_dataset import Phi3Dataset

        processor = AutoProcessor.from_pretrained(
            cfg["model"]["language_model"], trust_remote_code=True
        )
        model = Phi3VLM(cfg)
        train_ds = Phi3Dataset(data_yaml, split="train", processor=processor)
        val_ds = Phi3Dataset(data_yaml, split="val", processor=processor)
        tokenizer = processor.tokenizer

    else:
        from models.small_vlm import SmallVLM
        from data.yolo_dataset import YOLOVLMDataset

        processor = AutoProcessor.from_pretrained(cfg["model"]["vision_encoder"])
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["language_model"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = SmallVLM(cfg)
        train_ds = YOLOVLMDataset(data_yaml, split="train", processor=processor, tokenizer=tokenizer)
        val_ds = YOLOVLMDataset(data_yaml, split="val", processor=processor, tokenizer=tokenizer)

    return model, train_ds, val_ds, processor, tokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Train Small VLM on a YOLO-format dataset")
    p.add_argument("--data", default=None, help="Path to dataset .yaml (default: auto-download COCO8)")
    p.add_argument("--config", default="configs/vlm_config.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides win over yaml
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    # Priority: --data CLI > dataset defined in config (auto-downloads on first run)
    data_yaml = args.data or ensure_dataset(cfg["dataset"])

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    model_type = cfg["model"].get("type", "custom")
    print(f"Device    : {device}")
    print(f"Model type: {model_type}")
    print(f"Dataset   : {data_yaml}")
    print(f"Epochs    : {cfg['training']['epochs']}")

    model, train_ds, val_ds, processor, tokenizer = build_components(cfg, data_yaml)
    if not cfg["model"].get("load_in_4bit", False):
        model = model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=2)

    optimizer = AdamW(trainable, lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    total_steps = len(train_loader) * cfg["training"]["epochs"]
    warmup_steps = cfg["training"].get("warmup_steps", 0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    save_path = Path(cfg["training"]["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)
    log_every = cfg["training"].get("log_every", 10)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            batch_device = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch_device)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if (step + 1) % log_every == 0:
                print(f"  step {step+1} loss={loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_device = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch_device)
                val_loss += outputs.loss.item()
        val_loss /= max(len(val_loader), 1)

        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}")

        metrics = evaluate_captions(model, val_ds, processor, tokenizer, device, model_type)
        print(format_metrics(metrics))

        ckpt = save_path / f"vlm_epoch{epoch:02d}.pt"
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "cfg": cfg}, ckpt)
        print(f"Saved: {ckpt}")


if __name__ == "__main__":
    main()

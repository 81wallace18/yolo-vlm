# Small VLM

A Small Vision-Language Model (VLM) trained on YOLO-format datasets. Given an image, the model generates a natural language description of the objects detected — e.g. `"A person at center. A car at bottom-left."`.

Two backends are available, selectable via config:

| Backend | Model | Trainable params | VRAM | Config |
|---|---|---|---|---|
| **custom** | CLIP ViT-B/32 + SmolLM-135M | ~2M (LoRA + Projection) | ~2 GB | `configs/vlm_config.yaml` |
| **phi3** | Phi-3-vision-128k-instruct | ~2M (LoRA) | ~3 GB (4-bit) / ~8 GB | `configs/phi3_config.yaml` |

---

## Architecture

### custom (default)

```
Image → CLIP ViT-B/32 (frozen) → Projection MLP → SmolLM-135M (LoRA) → Text
          87M params                  ~1M params        135M / ~2M trainable
```

| Component | Model | Params | Trainable |
|---|---|---|---|
| Vision Encoder | `openai/clip-vit-base-patch32` | 87M | No (frozen) |
| Projection MLP | Linear(768→768→576) + GELU | ~1M | Yes |
| Language Decoder | `HuggingFaceTB/SmolLM-135M` + LoRA (`q/k/v/o_proj`) | 135M | ~1M (LoRA) |

### phi3

```
Image + Prompt → Phi-3-vision-128k-instruct (LoRA on qkv_proj, o_proj) → Text
                        4.2B params — ~2M trainable (LoRA)
```

Phi-3-vision is a complete VLM — it handles vision encoding internally. No external CLIP or Projection MLP needed. Fine-tuning applies LoRA only on the attention layers, keeping the rest frozen.

| Component | Details | Trainable |
|---|---|---|
| Vision Encoder | Built-in (CLIP ViT-L/14) | No (frozen) |
| Language Model | Phi-3 4.2B + LoRA (`qkv_proj`, `o_proj`) | ~2M (LoRA) |
| 4-bit quantization | NF4 via bitsandbytes (optional) | — |

---

## Dataset Format

Both backends accept any dataset in [YOLO format](https://docs.ultralytics.com/datasets/detect/coco8/):

```
my_dataset/
├── images/
│   ├── train/   *.jpg / *.png
│   └── val/
└── labels/
    ├── train/   *.txt  — one line per object: class_id cx cy w h (normalized 0–1)
    └── val/
```

With a YAML descriptor:

```yaml
path: my_dataset
train: images/train
val:   images/val
names:
  0: person
  1: car
  ...
```

Bounding box annotations are converted to natural language captions at load time:

```
0 0.50 0.40 0.20 0.58   →   "A person at center."
2 0.10 0.80 0.15 0.30   →   "A car at bottom-left."
```

Position is mapped from normalized `(cx, cy)` using a 3×3 grid:
`top-left`, `top`, `top-right`, `left`, `center`, `right`, `bottom-left`, `bottom`, `bottom-right`.

---

## Quick Start

```bash
# Train custom backend on COCO8 (auto-downloaded on first run)
.venv/bin/python train.py

# Train Phi-3-vision on COCO8
.venv/bin/python train.py --config configs/phi3_config.yaml

# Train on a custom YOLO dataset
.venv/bin/python train.py --data /path/to/dataset.yaml

# Override hyperparameters without editing the config
.venv/bin/python train.py --epochs 5 --lr 2e-4 --batch-size 4

# Run inference — backend is auto-detected from the checkpoint
.venv/bin/python inference.py /path/to/image.jpg

# Run inference with a specific checkpoint
.venv/bin/python inference.py /path/to/image.jpg --checkpoint checkpoints/vlm_epoch05.pt
```

---

## Configuration

The `dataset` block in every config defines what to train on. The dataset is downloaded automatically on first run and cached at `~/.cache/yolo-vlm/`.

### custom — `configs/vlm_config.yaml`

```yaml
model:
  type: custom
  vision_encoder: "openai/clip-vit-base-patch32"
  vision_dim: 768
  projection_hidden_dim: 768
  language_model: "HuggingFaceTB/SmolLM-135M"
  language_dim: 576
  lora_rank: 8
  lora_alpha: 16
  max_new_tokens: 64

dataset:
  name: coco8
  download: "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"

training:
  epochs: 10
  batch_size: 8
  lr: 1e-4
  warmup_steps: 50
  device: "cuda"
  save_path: "checkpoints/"
```

### phi3 — `configs/phi3_config.yaml`

```yaml
model:
  type: phi3
  language_model: "microsoft/Phi-3-vision-128k-instruct"
  lora_rank: 8
  lora_alpha: 16
  load_in_4bit: true   # set false if you have >=8GB VRAM free
  max_new_tokens: 64

dataset:
  name: coco8
  download: "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"

training:
  epochs: 5
  batch_size: 2        # smaller — 4.2B model
  lr: 5e-5
  warmup_steps: 20
  device: "cuda"
  save_path: "checkpoints/"
```

To use a different dataset, update `dataset.name` and `dataset.download`, or pass `--data` at the CLI.

---

## Project Structure

```
.
├── configs/
│   ├── vlm_config.yaml       # custom backend: CLIP + SmolLM
│   └── phi3_config.yaml      # phi3 backend: Phi-3-vision-128k-instruct
├── data/
│   ├── download.py           # auto-downloads datasets from config URL
│   ├── yolo_dataset.py       # YOLO images + labels → (pixel_values, caption) [custom]
│   ├── phi3_dataset.py       # YOLO images + labels → Phi-3 chat format [phi3]
│   └── label_to_caption.py   # converts bbox annotations to natural language
├── models/
│   ├── vision_encoder.py     # frozen CLIP wrapper [custom]
│   ├── projection.py         # MLP bridging vision → language space [custom]
│   ├── language_decoder.py   # SmolLM-135M with LoRA [custom]
│   ├── small_vlm.py          # full custom model: encoder + projection + decoder
│   └── phi3_vlm.py           # Phi-3-vision wrapper with LoRA + optional 4-bit
├── train.py                  # training loop (works for both backends)
├── inference.py              # generate a caption (auto-detects backend from checkpoint)
└── requirements.txt
```

---

## Requirements

```
torch>=2.0
transformers>=4.40
peft>=0.10
Pillow>=9.0
PyYAML>=6.0
tqdm>=4.65
bitsandbytes>=0.41   # required only for phi3 with load_in_4bit: true
```

Install with:

```bash
pip install -r requirements.txt
```

---

## How It Works

### custom backend

1. **Vision encoding** — the image is passed through a frozen CLIP ViT-B/32, producing a 768-dim feature vector.
2. **Projection** — a 2-layer MLP maps the vision feature into SmolLM's embedding space (576-dim).
3. **Language decoding** — the projected vision token is prepended to the tokenized caption and fed into SmolLM-135M with LoRA applied to `q/k/v/o_proj`.
4. **Training objective** — causal language modeling loss (cross-entropy) on caption tokens only; the vision prefix position is masked with `-100`.

### phi3 backend

1. **Image + prompt** — image and a fixed prompt (`"Describe the objects in this image."`) are processed together by Phi-3's built-in processor using the `<|image_1|>` token.
2. **Single-model decoding** — Phi-3-vision handles vision and language in one forward pass; no external encoder or projection layer.
3. **LoRA fine-tuning** — adapters applied to `qkv_proj` and `o_proj`; everything else stays frozen.
4. **Training objective** — same causal LM loss, but prompt tokens are masked with `-100` so loss only covers the generated caption.
5. **4-bit quantization** — when `load_in_4bit: true`, weights are loaded in NF4 via bitsandbytes, reducing VRAM from ~8 GB to ~3 GB with minimal quality loss.

---

## Checkpoints

Checkpoints are saved to `checkpoints/vlm_epoch<N>.pt` after every epoch. Each checkpoint stores the model weights and the full config, so any run is reproducible and `inference.py` auto-detects the backend without extra flags:

```python
ckpt = torch.load("checkpoints/vlm_epoch05.pt")
cfg  = ckpt["cfg"]                    # exact config used during training
backend = cfg["model"]["type"]        # "custom" or "phi3"
```

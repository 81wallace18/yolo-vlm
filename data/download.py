import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

CACHE_DIR = Path.home() / ".cache" / "yolo-vlm"


def ensure_dataset(dataset_cfg: dict) -> Path:
    """Download and extract a dataset if not already cached.

    Reads `name` and `download` from the dataset block in vlm_config.yaml.
    Returns the path to the local <name>.yaml file.

    Example config block:
        dataset:
          name: coco8
          download: "https://...coco8.zip"
    """
    name = dataset_cfg["name"]
    url = dataset_cfg["download"]

    dest = CACHE_DIR / name
    yaml_path = dest / f"{name}.yaml"

    if yaml_path.exists():
        return yaml_path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / f"{name}.zip"

    print(f"Downloading {name} dataset...")
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=f"{name}.zip") as bar:
        def _hook(count, block_size, total):
            if bar.total is None and total > 0:
                bar.total = total
            bar.update(block_size)
        urllib.request.urlretrieve(url, zip_path, reporthook=_hook)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(CACHE_DIR)
    zip_path.unlink()

    _write_yaml(name, dest, yaml_path)
    print(f"{name} ready at {dest}")
    return yaml_path


def _write_yaml(name: str, root: Path, yaml_path: Path):
    """Generate a <name>.yaml pointing at the extracted dataset."""
    if name == "coco8":
        _write_coco8_yaml(root, yaml_path)
    else:
        # Generic fallback: minimal yaml pointing at the extracted folder
        yaml_path.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames: {{}}\n")


def _write_coco8_yaml(root: Path, yaml_path: Path):
    names = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
        10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
        14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
        20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
        25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
        30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
        34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
        38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup",
        42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana",
        47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot",
        52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
        57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
        61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
        66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
        70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
        75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
        79: "toothbrush",
    }
    lines = [f"path: {root}\n", "train: images/train\n", "val: images/val\n", "names:\n"]
    lines += [f"  {k}: {v}\n" for k, v in names.items()]
    yaml_path.write_text("".join(lines))

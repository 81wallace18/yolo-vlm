def _position(cx: float, cy: float) -> str:
    col = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")
    row = "top" if cy < 0.33 else ("bottom" if cy > 0.66 else "center")
    if row == "center" and col == "center":
        return "center"
    if row == "center":
        return col
    if col == "center":
        return row
    return f"{row}-{col}"


def yolo_labels_to_caption(label_path: str, class_names: dict) -> str:
    """Convert a YOLO .txt label file to a natural language caption.

    Args:
        label_path: path to .txt file with lines 'class_id cx cy w h'
        class_names: dict mapping int id → class name string

    Returns:
        Caption string, e.g. "A person at center. A car at bottom-left."
        Returns "An image with no labeled objects." if the file is empty.
    """
    try:
        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]
    except FileNotFoundError:
        return "An image with no labeled objects."

    if not lines:
        return "An image with no labeled objects."

    phrases = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        cx, cy = float(parts[1]), float(parts[2])
        name = class_names.get(class_id, f"object_{class_id}")
        pos = _position(cx, cy)
        article = "An" if name[0].lower() in "aeiou" else "A"
        phrases.append(f"{article} {name} at {pos}.")

    return " ".join(phrases) if phrases else "An image with no labeled objects."

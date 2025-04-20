from ultralytics import YOLO
from typing import Dict

def load_models() -> Dict[str, YOLO]:
    """Load all YOLO model variants."""
    return {
        'nano': YOLO("yolo11n.pt"),
        'small': YOLO("yolo11s.pt"),
        'medium': YOLO("yolo11m.pt"),
        'large': YOLO("yolo11l.pt"),
        'xlarge': YOLO("yolo11x.pt")
    }

def get_model(model_path: str) -> YOLO:
    """Load a specific model from path."""
    return YOLO(model_path)

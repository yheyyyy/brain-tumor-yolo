import cv2
import matplotlib.pyplot as plt
from .models import get_model
from .visualization import plot_comparison

class ModelEvaluator:
    def __init__(self, model_path: str):
        self.model = get_model(model_path)
    
    def evaluate(self, validation_pairs):
        """Evaluate model on validation images."""
        for img_path, label_path in validation_pairs:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = self.model(img_path)
            plot_comparison(image, results[0].plot(), img_path)

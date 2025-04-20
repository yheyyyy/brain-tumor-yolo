import os
import cv2
import torch
import locale
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

class BrainTumorDetector:
    def __init__(self, config):
        """Initialize the detector with configuration."""
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        locale.getpreferredencoding = lambda: "UTF-8"
        
        # Initialize models
        self.models = {
            'nano': YOLO("yolo11n.pt"),
            'small': YOLO("yolo11s.pt"),
            'medium': YOLO("yolo11m.pt"),
            'large': YOLO("yolo11l.pt"),
            'xlarge': YOLO("yolo11x.pt")
        }
        
    def load_data(self):
        """Download and prepare the dataset."""
        os.system('wget -q https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/brain-tumor.yaml -O brain-tumor.yaml')
        self.data_path = Path('brain-tumor.yaml')
        
    def train_model(self, model_size='nano', save_dir='runs/train'):
        """Train the model with specified configuration."""
        model = self.models[model_size]
        
        training_args = {
            'data': str(self.data_path),
            'epochs': self.config['epochs'],
            'batch': self.config['batch_size'],
            'imgsz': self.config['image_size'],
            'save_dir': save_dir,
            'device': self.device,
            # Data augmentation parameters
            'scale': self.config['augmentation']['scale'],
            'mosaic': self.config['augmentation']['mosaic'],
            'mixup': self.config['augmentation']['mixup'],
            'copy_paste': self.config['augmentation']['copy_paste'],
            'degrees': self.config['augmentation']['rotation'],
            'hsv_h': self.config['augmentation']['hsv_h'],
            'hsv_s': self.config['augmentation']['hsv_s'],
            'hsv_v': self.config['augmentation']['hsv_v']
        }
        
        results = model.train(**training_args)
        return results
        
    def evaluate_model(self, model_path, validation_images):
        """Evaluate the trained model on validation images."""
        model = YOLO(model_path)
        
        for img_path, label_path in validation_images:
            # Load and process image
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get predictions
            results = model(img_path)
            
            # Visualize results
            self._plot_comparison(image, results[0].plot(), img_path)
            
    def _plot_comparison(self, original, prediction, title):
        """Helper method to plot original vs prediction."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(original)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.imshow(prediction)
        ax2.set_title('Model Prediction')
        ax2.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

def main():
    # Configuration
    config = {
        'epochs': 10,
        'batch_size': 16,
        'image_size': 320,
        'augmentation': {
            'scale': 0.9,
            'mosaic': 0.9,
            'mixup': 0.2,
            'copy_paste': 0.4,
            'rotation': 10,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4
        }
    }
    
    # Initialize detector
    detector = BrainTumorDetector(config)
    
    # Load data
    detector.load_data()
    
    # Train model
    results = detector.train_model(
        model_size='nano',
        save_dir='runs/brain_tumor_detection'
    )
    
    # Validation image pairs
    validation_images = [
        ("./datasets/brain-tumor/valid/images/val_1 (1).jpg",
         "../datasets/brain-tumor/valid/labels/val_1 (1).txt"),
        ("./datasets/brain-tumor/valid/images/val_1 (2).jpg",
         "./datasets/brain-tumor/valid/labels/val_1 (2).txt"),
        ("./datasets/brain-tumor/valid/images/val_1 (3).jpg",
         "./datasets/brain-tumor/valid/labels/val_1 (3).txt")
    ]
    # Evaluate model
    best_model_path = Path('runs/detect/train/weights/best.pt')
    detector.evaluate_model(best_model_path, validation_images)

if __name__ == "__main__":
    main()

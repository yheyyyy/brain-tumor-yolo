from pathlib import Path
from .models import load_models
from .utils import load_config

class ModelTrainer:
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config = load_config(config_path)
        self.models = load_models()
    
    def train(self):
        """Train the model using configuration settings."""
        model = self.models[self.config['model']['size']]
        
        training_args = {
            'data': self.config['data']['local_path'],
            'epochs': self.config['training']['epochs'],
            'batch': self.config['training']['batch_size'],
            'imgsz': self.config['training']['image_size'],
            'save_dir': self.config['model']['save_dir'],
            'device': self.config['training']['device'],
            **self.config['augmentation']
        }
        
        return model.train(**training_args)

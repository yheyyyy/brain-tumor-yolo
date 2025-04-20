import click
from pathlib import Path
from .trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .utils import download_dataset

@click.command()
@click.option('--config', default='config/default.yaml', help='Path to config file')
def main(config):
    """Main training pipeline."""
    # Initialize
    trainer = ModelTrainer(config)
    
    # Download dataset
    download_dataset()
    
    # Train model
    trainer.train()
    
    # Evaluate
    best_model_path = Path(trainer.config['model']['save_dir']) / 'weights/best.pt'
    evaluator = ModelEvaluator(str(best_model_path))
    evaluator.evaluate(trainer.config['data']['validation_images'])

if __name__ == "__main__":
    main()

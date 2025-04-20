# Brain Tumor Detection using YOLOv11

This project implements a brain tumor detection system using YOLOv11 object detection models. It provides a complete pipeline for training and evaluating models on medical imaging data.

## Features

- Support for multiple YOLO model architectures (nano to xlarge)
- Configurable data augmentation pipeline
- Training progress visualization
- Model evaluation tools
- Validation on medical images

## Requirements

```bash
pip install ultralytics==8.3.107
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Usage

1. Clone the repository
2. Install dependencies
3. Run the training script:

```bash
python train.py
```

## Configuration

The training configuration can be modified in the `config` dictionary within `train.py`. Key parameters include:

- Epochs
- Batch size
- Image size
- Data augmentation settings
- Model architecture selection

## Model Architectures

Available YOLO models:
- YOLOv11n (nano)
- YOLOv11s (small)
- YOLOv11m (medium)
- YOLOv11l (large)
- YOLOv11x (xlarge)

## Results

Training results and model weights are saved in the `runs/brain_tumor_detection` directory

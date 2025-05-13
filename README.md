# Image Classification with Transfer Learning

This project demonstrates image classification using transfer learning with a pre-trained ResNet-18 model on the CIFAR-10 dataset.

## Project Structure

- `main.py`: Script for training the model
- `test_model.py`: Script for testing the trained model on new images
- `requirements.txt`: Dependencies for the project

## Setup and Installation

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the training script:
```
python main.py
```

This will:
- Download the CIFAR-10 dataset
- Train a ResNet-18 model using transfer learning
- Save the trained model as `cifar10_resnet18.pth`
- Generate training metrics plots

### Testing on New Images

To test the model on your own images:
```
python test_model.py
```

You will be prompted to enter the path to an image for classification.

## CIFAR-10 Classes

The model classifies images into the following 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Implementation Details

- **Architecture**: ResNet-18 with modified fully connected layer
- **Transfer Learning**: Pre-trained weights on ImageNet dataset
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss
- **Training**: 10 epochs by default
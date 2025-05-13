import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torchvision.models as models

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Classes in CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Transformation for input images - must match the one used in training
transform = transforms.Compose([
    transforms.Resize(64),  # Make sure this matches your training resize
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load the saved model
def load_model(model_path):
    # Initialize model architecture - must match what you trained
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)  # CIFAR-10 has 10 classes
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

# Function to predict a single image
def predict_image(image_path, model):
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        print(f"Image loaded successfully: {image_path}")
        
        # Apply transform
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Get prediction and confidence
        predicted_class = classes[predicted.item()]
        confidence = probabilities[predicted.item()].item()
        
        # Create a bar plot of class probabilities
        plt.figure(figsize=(10, 5))
        
        # Plot the image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}")
        plt.axis('off')
        
        # Plot the probabilities
        plt.subplot(1, 2, 2)
        probs = probabilities.cpu().numpy()
        y_pos = np.arange(len(classes))
        plt.barh(y_pos, probs)
        plt.yticks(y_pos, classes)
        plt.xlabel('Probability')
        plt.title('Class Probabilities')
        
        plt.tight_layout()
        plt.show()
        
        return predicted_class, confidence, probabilities.cpu().numpy()
    
    except Exception as e:
        print(f"Error processing image: {e}")
        raise e

# Main function
if __name__ == "__main__":
    # Path to model
    model_path = 'outputs/models/cifar10_mobilenet.pth'
    
    try:
        # Load the model
        print("Loading model...")
        model = load_model(model_path)
        print("Model loaded successfully!")
        
        # Get image path from user
        image_path = input("Enter the path to the image you want to classify: ")
        
        # Make prediction
        predicted_class, confidence, _ = predict_image(image_path, model)
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Make sure you've trained the model first.")
    except Exception as e:
        print(f"An error occurred: {e}")
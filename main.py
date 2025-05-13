import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
import time
import os

# Create directories if they don't exist
os.makedirs('outputs/models', exist_ok=True)
os.makedirs('outputs/visualizations', exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters - reduced for faster training
num_epochs = 3  # Reduced from 10 to 3
batch_size = 32  # Reduced from 64 to 32
learning_rate = 0.001

# Data transformations - simplified for faster processing
transform_train = transforms.Compose([
    transforms.Resize(64),  # Reduced from 224 to 64
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.Resize(64),  # Reduced from 224 to 64
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

print("Loading CIFAR-10 dataset...")
# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True, 
    transform=transform_train
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True, 
    transform=transform_test
)

# Use smaller subsets for faster training
train_subset_size = 5000  # Using only 5000 samples for training
test_subset_size = 1000   # Using only 1000 samples for testing

# Create subsets
train_indices = torch.randperm(len(train_dataset))[:train_subset_size]
test_indices = torch.randperm(len(test_dataset))[:test_subset_size]

train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

# Create data loaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

# Define the classes in CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("Creating model...")
# Use a smaller model: MobileNetV2 instead of ResNet
model = models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)

# Modify the final fully connected layer for 10 classes (CIFAR-10)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)

# Move model to device
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to train the model
def train_model():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (i+1) % 20 == 0:  # Print more frequently for feedback
            elapsed = time.time() - start_time
            print(f'Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Time: {elapsed:.2f}s')
            start_time = time.time()  # Reset timer
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# Function to evaluate the model
def evaluate_model():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    return test_loss, test_acc

# Plot training and validation metrics
def plot_metrics(train_losses, train_accs, test_losses, test_accs):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(test_accs, label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/visualizations/training_metrics.png')
    print("Training metrics plot saved to 'outputs/visualizations/training_metrics.png'")
    plt.show()

# Function to visualize some predictions
def visualize_predictions():
    model.eval()
    images, labels = next(iter(test_loader))
    
    # Get predictions
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    # Convert tensors to numpy
    images = images.cpu().numpy()
    labels = labels.numpy()
    predicted = predicted.cpu().numpy()
    
    # Display images and predictions
    fig = plt.figure(figsize=(12, 5))
    num_images = min(8, len(images))  # Show up to 8 images
    
    for i in range(num_images):
        ax = fig.add_subplot(2, 4, i+1, xticks=[], yticks=[])
        # Denormalize and reshape the image
        img = np.transpose(images[i], (1, 2, 0))
        img = img * np.array((0.2023, 0.1994, 0.2010)) + np.array((0.4914, 0.4822, 0.4465))
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f"True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}")
    
    plt.tight_layout()
    plt.savefig('outputs/visualizations/predictions.png')
    print("Predictions plot saved to 'outputs/visualizations/predictions.png'")
    plt.show()

# Training loop
if __name__ == "__main__":
    # Start the timer
    total_start_time = time.time()
    
    # Initialize lists to store metrics
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        
        # Train the model
        train_loss, train_acc = train_model()
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate the model
        test_loss, test_acc = evaluate_model()
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        epoch_time = time.time() - epoch_start_time
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'Epoch completed in {epoch_time:.2f} seconds')
        print('-----------------------------------')
    
    # Calculate total training time
    total_time = time.time() - total_start_time
    print(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    # Plot metrics
    plot_metrics(train_losses, train_accs, test_losses, test_accs)

    # Save the model
    model_path = 'outputs/models/cifar10_mobilenet.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to '{model_path}'")

    # Visualize some predictions
    visualize_predictions()
    
    print("Training and evaluation complete!")
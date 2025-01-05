import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time
import warnings
from torch.utils.data import DataLoader, Subset

# Ignore all warnings
warnings.filterwarnings("ignore")

# Define the hyperparameters and data loaders
batch_size = 128
learning_rate = 0.001
num_epochs = 10

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = CIFAR10(root='./data', train=True, transform=train_transform, download=True)
subset_indices = list(range(20))
subset_train_dataset = Subset(train_dataset, subset_indices)

# Define the DataLoader with the subset
batch_size = 4  # Example batch size
train_loader = DataLoader(subset_train_dataset, batch_size=batch_size, shuffle=True)

# Create the model (MobileNetV2)
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)

# Modify the final fully connected layer for CIFAR-10 (10 classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Record the overall training start time
start_time = time.time()

# Train the model
print('Training started')
progress_bar = tqdm(range(num_epochs))
for epoch in range(num_epochs):
    epoch_start_time = time.time()  # Track the start time of the epoch
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update progress bar after each batch
        progress_bar.set_postfix(train_loss=running_loss / (i + 1))
        progress_bar.update(1)

    # Step the scheduler after each epoch
    scheduler.step()

    epoch_end_time = time.time()  # Track the end time of the epoch
    epoch_duration = epoch_end_time - epoch_start_time  # Calculate epoch duration

    # Print statistics
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader):.4f}, "
          f"Epoch Time: {epoch_duration:.2f} seconds")

# Record the total training time
total_training_time = time.time() - start_time
print(f"\nTraining complete. Total time: {total_training_time:.2f} seconds.")

# Saving model
print("Saving model...")
torch.save(model.state_dict(), "mobilenet_v2_model.pth")

# Test the model
model.eval()  # Set the model to evaluation mode
test_dataset = CIFAR10(root='./data', train=False, transform=test_transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test accuracy: {accuracy:.2f}%")

print('\nFinished training')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import init_process_group, barrier
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.pipeline.sync import Pipe
from torchvision.models import mobilenet_v2
import os
import time
import warnings
from tqdm.auto import tqdm

# Ignore warnings
warnings.filterwarnings("ignore")

# Distributed training setup
def init_distributed_mode():
    init_process_group(
        backend="nccl",
        init_method="env://",  # Default method for multi-node
        world_size=int(os.environ['WORLD_SIZE']),
        rank=int(os.environ["RANK"])
    )
    barrier()

# Set up the device for each process
def setup_device():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return torch.device(f'cuda:{local_rank}')

# Initialize distributed mode and set device
init_distributed_mode()
device = setup_device()

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Dataset preparation
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
test_dataset = CIFAR10(root='./data', train=False, transform=test_transform, download=True)

# Use DistributedSampler to split the dataset across processes
train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# MobileNetV2 model with layer parallelism
class MobileNetV2Stages(nn.Module):
    def __init__(self):
        super(MobileNetV2Stages, self).__init__()
        mobilenet = mobilenet_v2(pretrained=False)
        # Modify the classifier for CIFAR-10 (10 classes)
        mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 10)
        self.stage1 = mobilenet.features[:25]  # First 25 layers
        self.stage2 = mobilenet.features[25:]  # Remaining layers
        self.stage3 = mobilenet.classifier    # Classifier

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x

# Create pipeline for layer parallelism
def create_pipeline():
    model = MobileNetV2Stages()
    # Assign stages to devices
    stages = [
        model.stage1.to("cuda:0"),
        model.stage2.to("cuda:1"),
        model.stage3.to("cuda:2")
    ]
    return Pipe(nn.Sequential(*stages), chunks=4)

pipeline_model = create_pipeline()

# Wrap the pipeline in DDP
ddp_model = DDP(pipeline_model, device_ids=[device])

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ddp_model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop
if torch.distributed.get_rank() == 0:
    print('Training started')

progress_bar = tqdm(range(num_epochs)) if torch.distributed.get_rank() == 0 else None

for epoch in range(num_epochs):
    ddp_model.train()
    train_sampler.set_epoch(epoch)
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if progress_bar:
            progress_bar.set_postfix(train_loss=running_loss / (i + 1))
            progress_bar.update(1)

    scheduler.step()

    if torch.distributed.get_rank() == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader):.4f}")

if torch.distributed.get_rank() == 0:
    print("Training complete.")

# Testing loop
ddp_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = ddp_model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

if torch.distributed.get_rank() == 0:
    print(f"Test accuracy: {100 * correct / total:.2f}%")

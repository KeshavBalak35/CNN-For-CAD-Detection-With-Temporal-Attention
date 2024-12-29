import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pydicom as pyd
from PIL import Image
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define download path
path_download = "salikhussaini49/sunnybrook-cardiac-mri"
dataset_path = kagglehub.datasets.dataset_download(path_download)

# Image processing function
def imageProcess(imgpath):
    img = pyd.dcmread(imgpath).pixel_array
    img = Image.fromarray(img).convert('RGB').resize((224, 224))
    return np.array(img)

# Explore the directory structure and process images
def explore_img_directory(path):
    img_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.dcm'):
                img_files.append(os.path.join(root, file))
    return img_files

img_files = explore_img_directory(dataset_path)
processed_image_files = [imageProcess(img) for img in img_files]

# Create labels (assuming first half are positive, second half are negative)
labels = [1] * (len(processed_image_files) // 2) + [0] * (len(processed_image_files) - len(processed_image_files) // 2)

# Custom Dataset
class ProcessedImageDataset(Dataset):
    def __init__(self, images, labels, transform=None, num_frames=10):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        # Simulate a sequence of frames by repeating the image
        sequence = [image] * self.num_frames
        
        if self.transform:
            sequence = [self.transform(Image.fromarray(frame)) for frame in sequence]
        
        sequence = torch.stack(sequence)
        return sequence, torch.tensor(label, dtype=torch.float32)

# Data augmentation and normalization
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Split data
train_images, test_images, train_labels, test_labels = train_test_split(
    processed_image_files, labels, test_size=0.2, random_state=42, stratify=labels
)

# Create datasets
train_dataset = ProcessedImageDataset(train_images, train_labels, transform=train_transform, num_frames=10)
test_dataset = ProcessedImageDataset(test_images, test_labels, transform=test_transform, num_frames=10)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the temporal attention mechanism
class TemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        weights = self.attention(x).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)

# Define the model with temporal attention
class CADDetector(nn.Module):
    def __init__(self, num_frames=10):
        super(CADDetector, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        self.temporal_attention = TemporalAttention(num_ftrs)
        
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b*t, c, h, w)
        features = self.resnet(x)
        features = features.view(b, t, -1)
        
        attended_features = self.temporal_attention(features)
        return self.fc(attended_features).squeeze()

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CADDetector(num_frames=10).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Training function
def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

# Evaluation function
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / len(dataloader.dataset), all_preds, all_labels

# Training loop
num_epochs = 50
best_val_loss = float('inf')
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_loss, val_preds, val_labels = evaluate(model, test_loader, criterion)
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_cad_detector_model.pth')
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    if epoch % 10 == 0:
        val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
        accuracy = accuracy_score(val_labels, val_preds_binary)
        precision = precision_score(val_labels, val_preds_binary)
        recall = recall_score(val_labels, val_preds_binary)
        auc = roc_auc_score(val_labels, val_preds)
        
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'AUC: {auc:.4f}')

# Final evaluation
model.load_state_dict(torch.load('best_cad_detector_model.pth'))
_, test_preds, test_labels = evaluate(model, test_loader, criterion)
test_preds_binary = (np.array(test_preds) > 0.5).astype(int)

accuracy = accuracy_score(test_labels, test_preds_binary)
precision = precision_score(test_labels, test_preds_binary)
recall = recall_score(test_labels, test_preds_binary)
auc = roc_auc_score(test_labels, test_preds)

print("Final Test Results:")
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'AUC: {auc:.4f}')

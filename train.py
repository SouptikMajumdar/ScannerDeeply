import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from models import ResNetModel, UNetResNet50
from GazeMapDataset import GazeMapDataset
import utils
import matplotlib.pyplot as plt
import random
from losses import kl_loss



# Define transformations
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

map_transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with mean=0.5 and std=0.5
])


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=40):
    best_val_loss = float('inf')
    best_model_weights = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, gaze_maps in train_loader:
            B, C, H, W = images.shape
            images = images.cuda()
            gaze_maps = gaze_maps.cuda()
            
            optimizer.zero_grad()
            outputs = model(images).reshape(B, 1, H, W)
            
            #outputs = F.log_softmax(outputs, dim=1)
            #gaze_maps = F.normalize(gaze_maps, p=1, dim=1)
            
            loss = criterion(outputs, gaze_maps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        val_loss = evaluate_model(model, val_loader, criterion)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()

    # Load the best model weights
    if best_model_weights:
        model.load_state_dict(best_model_weights)
        torch.save(model.state_dict(), 'models/UNetRes50.pth')

def main():
    # Directories for train and validation
    train_images_dir = 'data/datasets_0221/correct/images/train'
    train_maps_dir = 'data/datasets_0221/correct/maps/train'
    val_images_dir = 'data/datasets_0221/correct/images/val'
    val_maps_dir = 'data/datasets_0221/correct/maps/val'
    # Create datasets
    train_dataset = GazeMapDataset(train_images_dir, train_maps_dir, image_transform=image_transform, map_transform=map_transform, split='train')
    val_dataset = GazeMapDataset(val_images_dir, val_maps_dir, image_transform=image_transform, map_transform=map_transform, split='val')
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 1e-4

    model = UNetResNet50(train_enc=False).cuda()
    #criterion = nn.KLDivLoss(reduction='batchmean')
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print('here')

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=40)


if __name__ == '__main__':
    main()

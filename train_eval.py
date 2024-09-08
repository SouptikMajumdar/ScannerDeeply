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

def evaluate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, gaze_maps in val_loader:
            B, C, H, W = images.shape
            images = images.cuda()
            gaze_maps = gaze_maps.cuda()
            outputs = model(images).reshape(B, 1, H, W)

            #Fix of KL Divergence:
            outputs = outputs.reshape(outputs.shape[0],-1)
            gaze_maps = gaze_maps.reshape(gaze_maps.shape[0],-1)

            outputs = F.log_softmax(outputs, dim=1)
            gaze_maps = F.softmax(gaze_maps, dim=1)

            loss = criterion(outputs, gaze_maps)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=40):
    best_val_loss = float('inf')
    best_model_weights = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        step=0
        for images, gaze_maps in train_loader:
            B, C, H, W = images.shape
            images = images.cuda()
            gaze_maps = gaze_maps.cuda()
            
            optimizer.zero_grad()
            outputs = model(images).reshape(B, 1, H, W)
            #Fix of KL Divergence:
            outputs = outputs.reshape(outputs.shape[0],-1)
            gaze_maps = gaze_maps.reshape(gaze_maps.shape[0],-1)
            outputs = F.log_softmax(outputs, dim=1)
            gaze_maps = F.softmax(gaze_maps, dim=1)

            
            #loss = criterion(torch.log(outputs_dist), (gaze_maps_dist))
            loss = criterion(outputs, gaze_maps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            step = step + 1
            print(f"Step [{step}], Train Loss: {loss.item():.4f}")
        
        avg_train_loss = running_loss / len(train_loader)
        val_loss = evaluate_model(model, val_loader, criterion)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save the best model
        #if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = model.state_dict()
        torch.save(model.state_dict(), 'models/UNetRes50.pth')

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
    

    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 1e-4

    model = UNetResNet50(train_enc=True).cuda()
    criterion = nn.KLDivLoss(reduction='batchmean', log_target=False)
    #criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=40)


if __name__ == '__main__':
    main()

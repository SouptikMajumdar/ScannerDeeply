import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import ResNetModel, UNetResNet50
from GazeMapDataset import GazeMapDataset

path = "models/UNetRes50.pth"

def visualize_random_result(model, val_loader):
    model.eval()

    data_iter = iter(val_loader)
    num_batches = len(val_loader)
    random_batch_idx = random.randint(0, num_batches - 1)

    for _ in range(random_batch_idx):
        images, gaze_maps = next(data_iter)
    
    batch_size = images.size(0)
    random_image_idx = random.randint(0, batch_size - 1)
    
    image = images[random_image_idx].unsqueeze(0).cuda()
    print(image.shape)
    true_gaze_map = gaze_maps[random_image_idx].unsqueeze(0)

    with torch.no_grad():
        predicted_gaze_map = model(image)

    image = image.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
    predicted_gaze_map = predicted_gaze_map.cpu().numpy().squeeze()
    true_gaze_map = true_gaze_map.cpu().numpy().squeeze()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image)
    axs[0].set_title('Input Image')
    axs[0].axis('off')

    axs[1].imshow(true_gaze_map, cmap='gray')
    axs[1].set_title('True Gaze Map')
    axs[1].axis('off')

    axs[2].imshow(predicted_gaze_map, cmap='gray')
    axs[2].set_title('Predicted Gaze Map')
    axs[2].axis('off')

    plt.show()

data_path = 'data/datasets_0221'

# Directories for train and validation
train_images_dir = f'{data_path}/correct/images/train'
train_maps_dir = f'{data_path}/correct/maps/train'
val_images_dir = f'{data_path}/correct/images/val'
val_maps_dir = f'{data_path}/correct/maps/val'

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

# Create datasets
train_dataset = GazeMapDataset(train_images_dir, train_maps_dir, image_transform=image_transform, map_transform=map_transform, split='train')
val_dataset = GazeMapDataset(val_images_dir, val_maps_dir, image_transform=image_transform, map_transform=map_transform, split='val')

batch_size = 16

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

path = "models/UNetRes50.pth"
state_dict = torch.load(path)
model = UNetResNet50().cuda() 
model.load_state_dict(state_dict)


visualize_random_result(model, train_loader)

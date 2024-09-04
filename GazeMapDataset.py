import os
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

class GazeMapDataset(Dataset):
    def __init__(self, images_dir, maps_dir, image_transform=None, map_transform=None, split='train'):
        self.images_dir = images_dir
        self.maps_dir = maps_dir
        self.image_transform = image_transform
        self.map_transform = map_transform


        # Find matching image and map files based on their suffixes
        self.image_filenames = []
        self.map_filenames = []



        image_files = [f for f in sorted(os.listdir(images_dir)) if not f.startswith(('.'))]
        map_files = [f for f in sorted(os.listdir(maps_dir)) if not f.startswith(('.'))]
        
        # Create sets for fast lookup
        image_suffixes = {os.path.splitext(f)[0].split('_')[-1] for f in image_files}
        map_suffixes = {os.path.splitext(f)[0].split('_')[-1] for f in map_files}

        # Find intersection of suffixes
        common_suffixes = image_suffixes.intersection(map_suffixes)

        # Filter filenames based on common suffixes
        for suffix in common_suffixes:
            if split == 'train':
                image_file = f'train_image_{suffix}.jpg'
                map_file = f'train_map_{suffix}.png'
            elif split == 'val':
                image_file = f'val_image_{suffix}.jpg'
                map_file = f'val_map_{suffix}.png'
            if image_file in image_files and map_file in map_files:
                self.image_filenames.append(image_file)
                self.map_filenames.append(map_file)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        map_path = os.path.join(self.maps_dir, self.map_filenames[idx])
        
        try:
            image = Image.open(img_path).convert("RGB")
            gaze_map = Image.open(map_path).convert("L")  # Grayscale (1 channel)
        except UnidentifiedImageError:
            raise ValueError(f"Cannot identify image file {img_path} or {map_path}")
        
        if self.image_transform:
            image = self.image_transform(image)
        if self.map_transform:
            gaze_map = self.map_transform(gaze_map)

        return image, gaze_map

import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from models import UNetResNet50

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="/netpool/homes/wangyo/Projects/chi2025_scanpath/evaluation/images/economist_daily_chart_85.png")
    parser.add_argument("--model_path", type=str, default="models/UNetRes50.pth")
    args = vars(parser.parse_args())
    state_dict = torch.load(args["model_path"])
    model = UNetResNet50().cuda() 
    model.load_state_dict(state_dict)

    model.eval()
    with Image.open(args["img_path"]).convert("RGB") as image:
        img_t = image_transform(image).unsqueeze(0).cuda()
        with torch.no_grad():
            heatmap = model(img_t).cpu().numpy().squeeze()
        heatmap = np.resize(heatmap, (image.size[1], image.size[0]))

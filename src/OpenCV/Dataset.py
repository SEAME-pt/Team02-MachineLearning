import os
import json
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

def get_binary_labels(height, width, pts, thickness=5):
    bin_img = np.zeros(shape=[height, width], dtype=np.uint8)
    for lane in pts:
        cv2.polylines(
            bin_img,
            np.int32([lane]),
            isClosed=False,
            color=1,
            thickness=thickness)
    return bin_img.astype(np.float32)[None, ...]   

def get_image_transform():
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    t = [transforms.ToTensor(),
         normalizer]

    transform = transforms.Compose(t)
    return transform

class TuSimpleDataset(Dataset):
    def __init__(self, json_paths, img_dir, width=512, height=256, 
                 thickness=5, max_lanes=4):
        self.samples = []
        self.width = width
        self.height = height
        self.thickness = thickness
        self.max_lanes = max_lanes
        self.img_dir = img_dir
        self.transform = get_image_transform()

        # Load all samples from JSON files
        for json_path in json_paths:
            with open(json_path, 'r') as f:
                for line in f:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        file_path = os.path.join(self.img_dir, info['raw_file'])
        
        # Read and resize image
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not read image: {file_path}")
            
        width_org = image.shape[1]
        height_org = image.shape[0]
        image = cv2.resize(image, (self.width, self.height))

        # Process lane points
        x_lanes = info['lanes']
        y_samples = info['h_samples']
        
        # Create points list with list comprehension
        pts = [
            [(x, y) for (x, y) in zip(lane, y_samples) if x >= 0]
            for lane in x_lanes
        ]

        # Remove empty lanes
        pts = [l for l in pts if len(l) > 0]

        # Calculate scaling rates
        x_rate = 1.0 * self.width / width_org
        y_rate = 1.0 * self.height / height_org

        # Scale points
        pts = [[(int(round(x*x_rate)), int(round(y*y_rate)))
                for (x, y) in lane] for lane in pts]

        # Generate labels
        bin_labels = get_binary_labels(self.height, self.width, pts,
                                    thickness=self.thickness)

        image = self.transform(image)

        bin_labels = torch.Tensor(bin_labels)

        return image, bin_labels
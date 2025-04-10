import os
import json
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from src.augmentation import LaneDetectionAugmentation
from torch.utils.data import Dataset

def get_image_transform():
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    t = [transforms.ToTensor(),
         normalizer]

    transform = transforms.Compose(t)
    return transform

class SEAMEDataset(Dataset):
    def __init__(self, img_dir, mask_dir, width=512, height=256, is_train=True):
        """
        SEA Dataset for lane detection
        
        Args:
            img_dir: Directory containing the images (frames)
            mask_dir: Directory containing the mask images
            width: Target image width
            height: Target image height
            is_train: Whether this is for training (enables augmentations)
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.width = width
        self.height = height
        self.transform = get_image_transform()
        self.is_train = is_train

        # Initialize augmentation for training
        self.augmentation = LaneDetectionAugmentation(
            height=height, 
            width=width,
        )

        self.image_files = sorted([f for f in os.listdir(img_dir)])
        
        # Create samples list - each sample is a tuple of (image_path, mask_path)
        # Only include images that have a corresponding mask
        self.samples = []
        valid_images = 0
        skipped_images = 0
        
        for img_file in self.image_files:
            img_base_name = os.path.splitext(img_file)[0]  # Get filename without extension
            
            mask_file = img_base_name + "_mask.png"
            mask_path = os.path.join(mask_dir, mask_file)
            
            if os.path.exists(mask_path):
                img_path = os.path.join(img_dir, img_file)
                self.samples.append((img_path, mask_path))
                valid_images += 1
            else:
                skipped_images += 1
            
        print(f"Dataset loaded: {valid_images} valid image-mask pairs, {skipped_images} images skipped due to missing masks")
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid image-mask pairs found. Check your directories: {img_dir} and {mask_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get image and mask paths from samples
        img_path, mask_path = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height))

        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        mask = cv2.resize(mask, (self.width, self.height))
        bin_labels = (mask > 127).astype(np.float32)[None, ...]
        
        if self.is_train:
            return self.augmentation(image, bin_labels)
        else:
            image = self.transform(image)
            return image, bin_labels
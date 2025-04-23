import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np

class LaneDetectionAugmentation:
    def __init__(self, height=256, width=512):
        self.height = height
        self.width = width
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.transform = A.Compose([
            # Resize to target resolution
            A.Resize(height=self.height, width=self.width),
            
            # Basic spatial transforms - increase rotation range for sharp turns
            A.HorizontalFlip(p=0.6),

            # This will move lanes left/right by up to 30% of image width
            A.OneOf([
                # Shift lanes heavily to the left
                A.Affine(
                    translate_percent={"x": (-0.35, -0.15), "y": (0, 0)},
                    scale=1.0,
                    rotate=0,
                    p=1.0
                ),
                # Shift lanes heavily to the right
                A.Affine(
                    translate_percent={"x": (0.15, 0.35), "y": (0, 0)},
                    scale=1.0,
                    rotate=0,
                    p=1.0
                ),
                # Center the lanes with variable position
                A.Affine(
                    translate_percent={"x": (-0.1, 0.1), "y": (0, 0)},
                    scale=(0.9, 1.1),  # Add some scaling variation 
                    rotate=(-5, 5),    # Add slight rotation
                    p=1.0
                ),
            ], p=0.8),

            A.Affine(scale=(0.95, 1.05), translate_percent=0.05, rotate=(-80, 80), p=0.5),
            
            # Moderate color transforms
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5)  # Subtle color shifts
            ], p=0.5),
            
            # Mild perspective transform to simulate camera angle changes
            A.Perspective(scale=(0.05, 0.15), keep_size=True, fit_output=True, p=0.5),
            
            # Light blur to simulate motion/focus issues
            A.OneOf([
                A.MotionBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
                A.GlassBlur(sigma=0.4, max_delta=2, iterations=1, p=0.2),  # Simulates reflections
            ], p=0.3),
            
            # Normalization and conversion to tensor
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
        
    def __call__(self, image, mask):
        """
        Apply transformations to image and mask
        
        Args:
            image: Input image (H, W, C) as numpy array
            mask: Binary mask (H, W) or (1, H, W) as numpy array
                
        Returns:
            transformed_image: Transformed image as tensor
            transformed_mask: Transformed mask as tensor
        """
        # If mask has a channel dimension, remove it for albumentations
        if len(mask.shape) == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)  # Remove channel dimension
        
        transformed = self.transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        
        # Ensure mask has channel dimension [1, H, W]
        if len(transformed_mask.shape) == 2:
            transformed_mask = transformed_mask.unsqueeze(0)
            
        return transformed_image, transformed_mask
    
class ObjectDetectionAugmentation:
    def __init__(self, height=192, width=384):
        self.transform = A.Compose([
            # Basic spatial augmentations
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            
            # Simpler spatial transforms that work across versions
            A.OneOf([
                A.Affine(scale=(0.9, 1.1), translate_percent=0.05, p=0.7),
                # Resize instead of crop (more stable API)
                A.Resize(height=height, width=width, p=1.0),
            ], p=0.5),
            
            # Color augmentations
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            
            # Simpler weather augmentations
            A.OneOf([
                A.RandomShadow(p=0.5),
                # A.RandomBrightness(p=0.5),
            ], p=0.3),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo',
            min_visibility=0.3,
            label_fields=['class_labels']
        ))
    
    def __call__(self, image, bboxes, class_labels):
        transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        return transformed
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
        ], additional_targets={'mask1': 'mask'})
        
    def __call__(self, image, binary_mask, instance_mask=None):
        """
        Apply transformations to image and masks
        
        Args:
            image: Input image (H, W, C) as numpy array
            binary_mask: Binary mask (H, W) or (1, H, W) as numpy array
            instance_mask: Instance segmentation mask (H, W) as numpy array
                
        Returns:
            transformed_image: Transformed image as tensor
            transformed_binary_mask: Transformed binary mask as tensor
            transformed_instance_mask: Transformed instance mask as tensor
        """
        # Remove channel dimension from masks if present
        if len(binary_mask.shape) == 3 and binary_mask.shape[0] == 1:
            binary_mask = binary_mask.squeeze(0)
        
        # Apply transformations to both masks
        if instance_mask is not None:
            transformed = self.transform(
                image=image, 
                mask=binary_mask,
                mask1=instance_mask
            )
            transformed_image = transformed['image']
            transformed_binary_mask = transformed['mask']
            transformed_instance_mask = transformed['mask1']
            
            # Ensure binary mask has channel dimension [1, H, W]
            if len(transformed_binary_mask.shape) == 2:
                transformed_binary_mask = transformed_binary_mask.unsqueeze(0)
                
            return transformed_image, transformed_binary_mask, transformed_instance_mask
        else:
            # Fallback to original behavior when no instance mask is provided
            transformed = self.transform(image=image, mask=binary_mask)
            transformed_image = transformed['image']
            transformed_binary_mask = transformed['mask']
            
            # Ensure mask has channel dimension [1, H, W]
            if len(transformed_binary_mask.shape) == 2:
                transformed_binary_mask = transformed_binary_mask.unsqueeze(0)
                
            return transformed_image, transformed_binary_mask
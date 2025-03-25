import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np

class LaneDetectionAugmentation:
    def __init__(self, height=256, width=512, is_train=True):
        self.height = height
        self.width = width
        self.is_train = is_train
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        if is_train:
            self.transform = A.Compose([
                # Resize to target resolution
                A.Resize(height=self.height, width=self.width),
                
                # Spatial transforms
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.95, 1.05), translate_percent=0.05, rotate=(-10, 10), p=0.5),
                
                # TUSimple-specific augmentations - focus on bottom portion of images
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.3),  # Shadows on road surface
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                                min_height=8, min_width=8, fill_value=0, p=0.3),
                
                # Color transforms
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                    A.RandomGamma(gamma_limit=(80, 120))
                ], p=0.5),
                
                # Weather simulation
                A.OneOf([
                    A.RandomRain(blur_value=3, p=0.5),
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.5),
                ], p=0.3),
                
                # Blur and noise
                A.OneOf([
                    A.MotionBlur(blur_limit=3),
                    A.MedianBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                    A.GaussNoise(var_limit=(10, 50)),
                ], p=0.3),
                
                # Normalization and conversion to tensor
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
        else:
            # Validation transformations - only normalize
            self.transform = A.Compose([
                A.Resize(height=self.height, width=self.width),  # Add this
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
    
    def __call__(self, image, mask):
        """
        Apply transformations to image and mask
        
        Args:
            image: Input image (H, W, C) as numpy array
            mask: Binary mask (1, H, W) as numpy array
            
        Returns:
            transformed_image: Transformed image as tensor
            transformed_mask: Transformed mask as tensor
        """
        # Remove channel dim from mask for albumentations
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)  # Shape becomes [H, W]
        
        # Apply transformations
        transformed = self.transform(image=image, mask=mask)
        
        # Get transformed image and mask
        transformed_image = transformed['image']  # Already a tensor from ToTensorV2
        
        # Check if mask is already a tensor or numpy array
        if isinstance(transformed['mask'], torch.Tensor):
            transformed_mask = transformed['mask'].float()
        else:
            transformed_mask = torch.from_numpy(transformed['mask']).float()
            
        # Add channel dimension back
        transformed_mask = transformed_mask.unsqueeze(0)  # Add channel dim back [1, H, W]
        
        return transformed_image, transformed_mask
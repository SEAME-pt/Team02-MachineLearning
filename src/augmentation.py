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
                
                # Basic spatial transforms - increase rotation range for sharp turns
                A.HorizontalFlip(p=0.5),

                A.Affine(scale=(0.95, 1.05), translate_percent=0.05, rotate=(-45, 45), p=0.5),
                
                # Simulate reflections on floor (specular highlights) and shadows
                A.OneOf([
                    A.RandomSunFlare(
                        flare_roi=(0.0, 0.0, 1.0, 0.5),
                        src_radius=40, 
                        src_color=(255, 255, 255),
                        p=0.3
                    ),
                    # Simulate reflections and glare (specific to your black reflective floor)
                    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.4),
                ], p=0.5),

                # Simulate shadows and occlusions (but fewer holes)
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.3),
                A.OneOf([
                    # White patches (could be glare/reflections)
                    A.CoarseDropout(
                        num_holes_range=(2, 4),
                        hole_height_range=(8, 20),
                        hole_width_range=(8, 20),
                        fill=255,
                        p=1.0
                    ),
                    # Dark patches (could be shadows/debris)
                    A.CoarseDropout(
                        num_holes_range=(2, 4),
                        hole_height_range=(8, 20),
                        hole_width_range=(8, 20),
                        fill=0,
                        p=1.0
                    ),
                    # Realistic inpainted patches (could be worn markings)
                    A.CoarseDropout(
                        num_holes_range=(2, 4),
                        hole_height_range=(8, 20),
                        hole_width_range=(8, 20),
                        fill="inpaint_ns",
                        p=1.0
                    ),
                ], p=0.3),
                
                # Moderate color transforms
                A.OneOf([
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.5),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5)  # Subtle color shifts
                ], p=0.5),

                # Floor division suppression
                A.OneOf([
                    # Reduce contrast in floor areas to minimize floor divisions
                    A.PixelDropout(dropout_prob=0.03, per_channel=True, drop_value=None, p=0.4),
                    # Smoothing to reduce impact of floor divisions
                    A.MedianBlur(blur_limit=3, p=0.4),
                ], p=0.5),

                # Enhance contrast to make lane markings more visible
                A.OneOf([
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.5),
                    A.Sharpen(alpha=(0.3, 0.6), lightness=(0.7, 1.0), p=0.5),
                ], p=0.4),
                
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
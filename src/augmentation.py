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
                    translate_percent={"x": (-0.05, -0.02), "y": (0, 0)},
                    scale=0.5,
                    rotate=0,
                    p=1.0
                ),
                # Shift lanes heavily to the right
                A.Affine(
                    translate_percent={"x": (0.02, 0.05), "y": (0, 0)},
                    scale=0.5,
                    rotate=0,
                    p=1.0
                ),
            ], p=0.5),

            # A.Affine(scale=(0.95, 1.05), translate_percent=0.05, rotate=(-80, 80), p=0.5),
            
            # Moderate color transforms
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5)  # Subtle color shifts
            ], p=0.5),
            
            # Mild perspective transform to simulate camera angle changes
            A.Perspective(scale=(0.03, 0.08), keep_size=True, fit_output=True, p=0.5),
            
            # Light blur to simulate motion/focus issues
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 3)),
                A.GaussianBlur(blur_limit=(3, 3)),
                A.GlassBlur(sigma=0.4, max_delta=2, iterations=1, p=0.2),  # Simulates reflections
            ], p=0.3),
            
            # Normalization and conversion to tensor
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ], additional_targets={'mask1': 'mask'})
        
    def __call__(self, image, binary_mask, instance_mask):
        """
        Apply augmentation to image and masks
        
        Args:
            image: RGB image (H, W, 3)
            binary_mask: Binary mask (2, H, W) 
            instance_mask: Instance segmentation mask (max_lanes, H, W)
        """
        # Transpose masks from CHW to HWC format for Albumentations
        binary_mask_transposed = binary_mask.transpose(1, 2, 0)  # (2, H, W) -> (H, W, 2)
        instance_mask_transposed = instance_mask.transpose(1, 2, 0)  # (4, H, W) -> (H, W, 4)
        
        # Apply transforms
        transformed = self.transform(
            image=image,
            mask=binary_mask_transposed,
            mask1=instance_mask_transposed
        )
        
        # Get the augmented data (already in tensor format from ToTensorV2)
        aug_image = transformed['image']  # Will be (3, H, W)
        
        # Convert masks back to CHW format for PyTorch
        if isinstance(transformed['mask'], np.ndarray):
            # For NumPy arrays
            aug_binary_mask = torch.from_numpy(transformed['mask'].transpose(2, 0, 1))
            aug_instance_mask = torch.from_numpy(transformed['mask1'].transpose(2, 0, 1))
        else:
            # For tensors (already transposed by ToTensorV2)
            aug_binary_mask = transformed['mask'].permute(2, 0, 1)
            aug_instance_mask = transformed['mask1'].permute(2, 0, 1)
        
        return aug_image, aug_binary_mask, aug_instance_mask
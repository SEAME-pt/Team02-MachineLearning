import torch
import random
from torch.utils.data import Dataset
from src.SEAMEDataset import SEAMEDataset
from src.TUSimpleDataset import TuSimpleDataset

class CombinedLaneDataset(Dataset):
    def __init__(self, tusimple_config, sea_config, val_split=0.2, seed=42):
        """
        Combined dataset that includes both TuSimple and SEA datasets
        with built-in train/validation split
        
        Args:
            tusimple_config: Dictionary with TuSimple dataset config
                {
                    'json_paths': [list of json annotation files],
                    'img_dir': 'path/to/tusimple/images',
                    'width': width,
                    'height': height,
                    'is_train': is_train,
                    'thickness': thickness
                }
            sea_config: Dictionary with SEA dataset config
                {
                    'img_dir': 'path/to/sea/images',
                    'mask_dir': 'path/to/sea/masks',
                    'width': width,
                    'height': height,
                    'is_train': is_train
                }
            val_split: Fraction of data to use for validation (default: 0.2)
            seed: Random seed for reproducible splits
        """
        self.val_split = val_split
        self.seed = seed
        random.seed(seed)
        
        # Create both datasets
        self.tusimple_dataset = TuSimpleDataset(
            json_paths=tusimple_config['json_paths'],
            img_dir=tusimple_config['img_dir'],
            width=tusimple_config.get('width', 512),
            height=tusimple_config.get('height', 256),
            is_train=tusimple_config.get('is_train', True),
            thickness=tusimple_config.get('thickness', 5)
        )
        
        self.sea_dataset = SEAMEDataset(
            img_dir=sea_config['img_dir'],
            mask_dir=sea_config['mask_dir'],
            width=sea_config.get('width', 512),
            height=sea_config.get('height', 256),
            is_train=sea_config.get('is_train', True)
        )
        
        # Store dataset sizes for indexing
        self.tusimple_size = len(self.tusimple_dataset)
        self.sea_size = len(self.sea_dataset)
        
        # Create indices for all samples
        self.tusimple_indices = list(range(self.tusimple_size))
        self.sea_indices = list(range(self.sea_size))
        
        # Shuffle indices
        random.shuffle(self.tusimple_indices)
        random.shuffle(self.sea_indices)
        
        # Split indices into train and validation
        tusimple_val_size = int(self.tusimple_size * self.val_split)
        sea_val_size = int(self.sea_size * self.val_split)
        
        self.tusimple_train_indices = self.tusimple_indices[tusimple_val_size:]
        self.tusimple_val_indices = self.tusimple_indices[:tusimple_val_size]
        
        self.sea_train_indices = self.sea_indices[sea_val_size:]
        self.sea_val_indices = self.sea_indices[:sea_val_size]
        
        # Store sizes for each split
        self.tusimple_train_size = len(self.tusimple_train_indices)
        self.sea_train_size = len(self.sea_train_indices)
        self.train_size = self.tusimple_train_size + self.sea_train_size
        
        self.tusimple_val_size = len(self.tusimple_val_indices)
        self.sea_val_size = len(self.sea_val_indices)
        self.val_size = self.tusimple_val_size + self.sea_val_size
        
        self.total_size = self.train_size + self.val_size
        
        print(f"Combined dataset created:")
        print(f"TuSimple: {self.tusimple_train_size} train, {self.tusimple_val_size} validation")
        print(f"SEA: {self.sea_train_size} train, {self.sea_val_size} validation")
        print(f"Total: {self.train_size} train, {self.val_size} validation")
        
        # Default to training mode
        self.is_validation = False
    
    def set_validation(self, is_validation=True):
        """Set whether to return validation or training samples"""
        self.is_validation = is_validation
        
        # Update dataset is_train flags
        if is_validation:
            # Disable augmentation for validation
            self.tusimple_dataset.is_train = False
            self.sea_dataset.is_train = False
        else:
            # Enable augmentation for training
            self.tusimple_dataset.is_train = True
            self.sea_dataset.is_train = True
        
        return self
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        if self.is_validation:
            return self.val_size
        else:
            return self.train_size
    
    def __getitem__(self, idx):
        """Get a sample from either training or validation set"""
        if self.is_validation:
            # Getting validation sample
            if idx < self.tusimple_val_size:
                # Get TuSimple validation sample
                actual_idx = self.tusimple_val_indices[idx]
                return self.tusimple_dataset[actual_idx]
            else:
                # Get SEA validation sample
                sea_idx = idx - self.tusimple_val_size
                # Add bounds check to prevent index errors
                if sea_idx >= len(self.sea_val_indices):
                    # If out of bounds, return a random sample from available indices
                    sea_idx = random.randint(0, len(self.sea_val_indices) - 1)
                actual_idx = self.sea_val_indices[sea_idx]
                return self.sea_dataset[actual_idx]
        else:
            # Getting training sample
            if idx < self.tusimple_train_size:
                # Get TuSimple training sample
                actual_idx = self.tusimple_train_indices[idx]
                return self.tusimple_dataset[actual_idx]
            else:
                # Get SEA training sample
                sea_idx = idx - self.tusimple_train_size
                # Add bounds check to prevent index errors
                if sea_idx >= len(self.sea_train_indices):
                    # If out of bounds, return a random sample from available indices
                    sea_idx = random.randint(0, len(self.sea_train_indices) - 1)
                actual_idx = self.sea_train_indices[sea_idx]
                return self.sea_dataset[actual_idx]

    def get_train_dataset(self):
        """Return a reference to this dataset in training mode"""
        return self.set_validation(False)
    
    def get_val_dataset(self):
        """Return a reference to this dataset in validation mode"""
        return self.set_validation(True)
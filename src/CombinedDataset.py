import torch
import random
from torch.utils.data import Dataset
from src.SEAMEDataset import SEAMEDataset
from src.TUSimpleDataset import TuSimpleDataset
from src.CarlaDataset import CarlaDataset

class CombinedLaneDataset(Dataset):
    def __init__(self, tusimple_config=None, sea_config=None, carla_config=None, val_split=0.2, seed=42):
        """
        Combined dataset that includes TuSimple, SEA, Carla, and BDD100K datasets
        with built-in train/validation split
        
        Args:
            tusimple_config: Dictionary with TuSimple dataset config or None to skip
            sea_config: Dictionary with SEA dataset config or None to skip
            carla_config: Dictionary with Carla dataset config or None to skip
            val_split: Fraction of data to use for validation (default: 0.2)
            seed: Random seed for reproducible splits
        """
        self.val_split = val_split
        self.seed = seed
        random.seed(seed)
        
        # Initialize dataset variables
        self.tusimple_dataset = None
        self.sea_dataset = None
        self.carla_dataset = None
        
        # Create datasets if configs are provided
        if tusimple_config:
            self.tusimple_dataset = TuSimpleDataset(
                json_paths=tusimple_config['json_paths'],
                img_dir=tusimple_config['img_dir'],
                width=tusimple_config.get('width', 512),
                height=tusimple_config.get('height', 256),
                is_train=tusimple_config.get('is_train', True),
                thickness=tusimple_config.get('thickness', 5)
            )
        
        if sea_config:
            self.sea_dataset = SEAMEDataset(
                json_paths=sea_config['json_paths'],
                img_dir=sea_config['img_dir'],
                width=sea_config.get('width', 512),
                height=sea_config.get('height', 256),
                is_train=sea_config.get('is_train', True),
                thickness=sea_config.get('thickness', 5)
            )
            
        if carla_config:
            self.carla_dataset = CarlaDataset(
                json_paths=carla_config['json_paths'],
                img_dir=carla_config['img_dir'],
                width=carla_config.get('width', 512),
                height=carla_config.get('height', 256),
                is_train=carla_config.get('is_train', True),
                thickness=carla_config.get('thickness', 5)
            )
        
        # Initialize sizes and indices
        self._initialize_dataset_indices()
        
        # Default to training mode
        self.is_validation = False
    
    def _initialize_dataset_indices(self):
        """Initialize all dataset indices and splits"""
        # Store dataset sizes for indexing
        self.tusimple_size = len(self.tusimple_dataset) if self.tusimple_dataset else 0
        self.sea_size = len(self.sea_dataset) if self.sea_dataset else 0
        self.carla_size = len(self.carla_dataset) if self.carla_dataset else 0
        
        # Create indices for all samples
        self.tusimple_indices = list(range(self.tusimple_size))
        self.sea_indices = list(range(self.sea_size))
        self.carla_indices = list(range(self.carla_size))
        
        # Shuffle indices
        if self.tusimple_size > 0:
            random.shuffle(self.tusimple_indices)
        if self.sea_size > 0:
            random.shuffle(self.sea_indices)
        if self.carla_size > 0:
            random.shuffle(self.carla_indices)
        
        # Split indices into train and validation
        tusimple_val_size = int(self.tusimple_size * self.val_split)
        sea_val_size = int(self.sea_size * self.val_split)
        carla_val_size = int(self.carla_size * self.val_split)
        
        # Create train/val index lists
        self.tusimple_train_indices = self.tusimple_indices[tusimple_val_size:] if self.tusimple_size > 0 else []
        self.tusimple_val_indices = self.tusimple_indices[:tusimple_val_size] if self.tusimple_size > 0 else []
        
        self.sea_train_indices = self.sea_indices[sea_val_size:] if self.sea_size > 0 else []
        self.sea_val_indices = self.sea_indices[:sea_val_size] if self.sea_size > 0 else []
        
        self.carla_train_indices = self.carla_indices[carla_val_size:] if self.carla_size > 0 else []
        self.carla_val_indices = self.carla_indices[:carla_val_size] if self.carla_size > 0 else []
        
        # Store sizes for each split
        self.tusimple_train_size = len(self.tusimple_train_indices)
        self.sea_train_size = len(self.sea_train_indices)
        self.carla_train_size = len(self.carla_train_indices)
        self.train_size = self.tusimple_train_size + self.sea_train_size + self.carla_train_size
        
        self.tusimple_val_size = len(self.tusimple_val_indices)
        self.sea_val_size = len(self.sea_val_indices)
        self.carla_val_size = len(self.carla_val_indices)
        self.val_size = self.tusimple_val_size + self.sea_val_size + self.carla_val_size
        
        self.total_size = self.train_size + self.val_size
        
        # Print dataset summary
        print(f"Combined dataset created:")
        if self.tusimple_size > 0:
            print(f"TuSimple: {self.tusimple_train_size} train, {self.tusimple_val_size} validation")
        if self.sea_size > 0:
            print(f"SEA: {self.sea_train_size} train, {self.sea_val_size} validation")
        if self.carla_size > 0:
            print(f"Carla: {self.carla_train_size} train, {self.carla_val_size} validation")
        print(f"Total: {self.train_size} train, {self.val_size} validation")
    
    def set_validation(self, is_validation=True):
        """Set whether to return validation or training samples"""
        self.is_validation = is_validation
        
        # Update dataset is_train flags
        if is_validation:
            # Disable augmentation for validation
            if self.tusimple_dataset:
                self.tusimple_dataset.is_train = False
            if self.sea_dataset:
                self.sea_dataset.is_train = False
            if self.carla_dataset:
                self.carla_dataset.is_train = False
        else:
            # Enable augmentation for training
            if self.tusimple_dataset:
                self.tusimple_dataset.is_train = True
            if self.sea_dataset:
                self.sea_dataset.is_train = True
            if self.carla_dataset:
                self.carla_dataset.is_train = True
        
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
            elif idx < self.tusimple_val_size + self.sea_val_size:
                # Get SEA validation sample
                sea_idx = idx - self.tusimple_val_size
                actual_idx = self.sea_val_indices[sea_idx]
                return self.sea_dataset[actual_idx]
            else:
                # Get Carla validation sample
                carla_idx = idx - self.tusimple_val_size - self.sea_val_size
                actual_idx = self.carla_val_indices[carla_idx]
                return self.carla_dataset[actual_idx]
        else:
            # Getting training sample
            if idx < self.tusimple_train_size:
                # Get TuSimple training sample
                actual_idx = self.tusimple_train_indices[idx]
                return self.tusimple_dataset[actual_idx]
            elif idx < self.tusimple_train_size + self.sea_train_size:
                # Get SEA training sample
                sea_idx = idx - self.tusimple_train_size
                actual_idx = self.sea_train_indices[sea_idx]
                return self.sea_dataset[actual_idx]
            else:
                # Get Carla training sample
                carla_idx = idx - self.tusimple_train_size - self.sea_train_size
                actual_idx = self.carla_train_indices[carla_idx]
                return self.carla_dataset[actual_idx]

    def get_train_dataset(self):
        """Return a reference to this dataset in training mode"""
        return self.set_validation(False)
    
    def get_val_dataset(self):
        """Return a reference to this dataset in validation mode"""
        return self.set_validation(True)
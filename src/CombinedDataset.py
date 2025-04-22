import torch
import random
from torch.utils.data import Dataset
from src.SEAMEDataset import SEAMEDataset
from src.CarlaDataset import CarlaDataset
from src.BDD100kDataset import BDD100KDataset  # Import BDD100K dataset

class CombinedLaneDataset(Dataset):
    def __init__(self, tusimple_config=None, sea_config=None, carla_config=None, bdd100k_config=None, val_split=0.2, seed=42):
        """
        Combined dataset that includes TuSimple, SEA, Carla, and BDD100K datasets
        with built-in train/validation split
        
        Args:
            tusimple_config: Dictionary with TuSimple dataset config or None to skip
            sea_config: Dictionary with SEA dataset config or None to skip
            carla_config: Dictionary with Carla dataset config or None to skip
            bdd100k_config: Dictionary with BDD100K dataset config or None to skip
            val_split: Fraction of data to use for validation (default: 0.2)
            seed: Random seed for reproducible splits
        """
        self.val_split = val_split
        self.seed = seed
        random.seed(seed)
        
        # Initialize dataset variables
        self.sea_dataset = None
        self.carla_dataset = None
        self.bdd100k_dataset = None
        
        if sea_config:
            self.sea_dataset = SEAMEDataset(
                img_dir=sea_config['img_dir'],
                mask_dir=sea_config['mask_dir'],
                width=sea_config.get('width', 512),
                height=sea_config.get('height', 256),
                is_train=sea_config.get('is_train', True)
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
        
        if bdd100k_config:
            self.bdd100k_dataset = BDD100KDataset(
                img_dir=bdd100k_config['img_dir'],
                labels_file=bdd100k_config['labels_file'],
                width=bdd100k_config.get('width', 512),
                height=bdd100k_config.get('height', 256),
                is_train=bdd100k_config.get('is_train', True),
                thickness=bdd100k_config.get('thickness', 5)
            )
        
        # Initialize sizes and indices
        self._initialize_dataset_indices()
        
        # Default to training mode
        self.is_validation = False
    
    def _initialize_dataset_indices(self):
        """Initialize all dataset indices and splits"""
        # Store dataset sizes for indexing
        self.sea_size = len(self.sea_dataset) if self.sea_dataset else 0
        self.carla_size = len(self.carla_dataset) if self.carla_dataset else 0
        self.bdd100k_size = len(self.bdd100k_dataset) if self.bdd100k_dataset else 0
        
        # Create indices for all samples
        self.sea_indices = list(range(self.sea_size))
        self.carla_indices = list(range(self.carla_size))
        self.bdd100k_indices = list(range(self.bdd100k_size))
        
        # Shuffle indices
        if self.sea_size > 0:
            random.shuffle(self.sea_indices)
        if self.carla_size > 0:
            random.shuffle(self.carla_indices)
        if self.bdd100k_size > 0:
            random.shuffle(self.bdd100k_indices)
        
        # Split indices into train and validation
        sea_val_size = int(self.sea_size * self.val_split)
        carla_val_size = int(self.carla_size * self.val_split)
        bdd100k_val_size = int(self.bdd100k_size * self.val_split)
        
        # Create train/val index lists
        
        self.sea_train_indices = self.sea_indices[sea_val_size:] if self.sea_size > 0 else []
        self.sea_val_indices = self.sea_indices[:sea_val_size] if self.sea_size > 0 else []
        
        self.carla_train_indices = self.carla_indices[carla_val_size:] if self.carla_size > 0 else []
        self.carla_val_indices = self.carla_indices[:carla_val_size] if self.carla_size > 0 else []
        
        self.bdd100k_train_indices = self.bdd100k_indices[bdd100k_val_size:] if self.bdd100k_size > 0 else []
        self.bdd100k_val_indices = self.bdd100k_indices[:bdd100k_val_size] if self.bdd100k_size > 0 else []
        
        # Store sizes for each split
        self.sea_train_size = len(self.sea_train_indices)
        self.carla_train_size = len(self.carla_train_indices)
        self.bdd100k_train_size = len(self.bdd100k_train_indices)
        self.train_size = self.tusimple_train_size + self.sea_train_size + self.carla_train_size + self.bdd100k_train_size
        
        self.sea_val_size = len(self.sea_val_indices)
        self.carla_val_size = len(self.carla_val_indices)
        self.bdd100k_val_size = len(self.bdd100k_val_indices)
        self.val_size = self.tusimple_val_size + self.sea_val_size + self.carla_val_size + self.bdd100k_val_size
        
        self.total_size = self.train_size + self.val_size
        
        # Print dataset summary
        print(f"Combined dataset created:")
        if self.sea_size > 0:
            print(f"SEA: {self.sea_train_size} train, {self.sea_val_size} validation")
        if self.carla_size > 0:
            print(f"Carla: {self.carla_train_size} train, {self.carla_val_size} validation")
        if self.bdd100k_size > 0:
            print(f"BDD100K: {self.bdd100k_train_size} train, {self.bdd100k_val_size} validation")
        print(f"Total: {self.train_size} train, {self.val_size} validation")
    
    def set_validation(self, is_validation=True):
        """Set whether to return validation or training samples"""
        self.is_validation = is_validation
        
        # Update dataset is_train flags
        if is_validation:
            # Disable augmentation for validation
            if self.sea_dataset:
                self.sea_dataset.is_train = False
            if self.carla_dataset:
                self.carla_dataset.is_train = False
            if self.bdd100k_dataset:
                self.bdd100k_dataset.is_train = False
        else:
            # Enable augmentation for training
            if self.sea_dataset:
                self.sea_dataset.is_train = True
            if self.carla_dataset:
                self.carla_dataset.is_train = True
            if self.bdd100k_dataset:
                self.bdd100k_dataset.is_train = True
        
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
            elif idx < self.tusimple_val_size + self.sea_val_size + self.carla_val_size:
                # Get Carla validation sample
                carla_idx = idx - self.tusimple_val_size - self.sea_val_size
                actual_idx = self.carla_val_indices[carla_idx]
                return self.carla_dataset[actual_idx]
            else:
                # Get BDD100K validation sample
                bdd100k_idx = idx - self.tusimple_val_size - self.sea_val_size - self.carla_val_size
                actual_idx = self.bdd100k_val_indices[bdd100k_idx]
                return self.bdd100k_dataset[actual_idx]
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
            elif idx < self.tusimple_train_size + self.sea_train_size + self.carla_train_size:
                # Get Carla training sample
                carla_idx = idx - self.tusimple_train_size - self.sea_train_size
                actual_idx = self.carla_train_indices[carla_idx]
                return self.carla_dataset[actual_idx]
            else:
                # Get BDD100K training sample
                bdd100k_idx = idx - self.tusimple_train_size - self.sea_train_size - self.carla_train_size
                actual_idx = self.bdd100k_train_indices[bdd100k_idx]
                return self.bdd100k_dataset[actual_idx]

    def get_train_dataset(self):
        """Return a reference to this dataset in training mode"""
        return self.set_validation(False)
    
    def get_val_dataset(self):
        """Return a reference to this dataset in validation mode"""
        return self.set_validation(True)
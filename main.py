import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from src.CombinedDataset import CombinedLaneDataset
from src.SEAMEDataset import SEAMEDataset
from src.train import train_model
from src.unet import UNet
import os
import numpy as np

def main():
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    "/home/luis_t2/SEAME/LaneAnchor/assets/TUSimple/train_set/label_data_0313.json",
    "/home/luis_t2/SEAME/LaneAnchor/assets/TUSimple/train_set/label_data_0601.json"

    # Your dataset configs
    tusimple_config = {
        'json_paths': ["/home/luis_t2/SEAME/LaneAnchor/assets/TUSimple/train_set/label_data_0313.json",
                        "/home/luis_t2/SEAME/LaneAnchor/assets/TUSimple/train_set/label_data_0531.json",
                        "/home/luis_t2/SEAME/LaneAnchor/assets/TUSimple/train_set/label_data_0601.json"],
        'img_dir': '/home/luis_t2/SEAME/LaneAnchor/assets/TUSimple/train_set/',
        'width': 256,
        'height': 128,
        'is_train': True,  # Set is_train=True for the base dataset
        'thickness': 3
    }
    
    sea_config = {
        'img_dir': '/home/luis_t2/SEAME/Dataset/frames',
        'mask_dir': '/home/luis_t2/SEAME/Dataset/masks',
        'width': 256,
        'height': 128,
        'is_train': True  # Set is_train=True for the base dataset
    }
    
    # Create the combined dataset with built-in train/val split
    combined_dataset = CombinedLaneDataset(tusimple_config, sea_config, val_split=0.0)
    
    # Get train and val datasets (these are views of the same dataset with different modes)
    train_dataset = combined_dataset.get_train_dataset()
    # val_dataset = combined_dataset.get_val_dataset()

    # Create weights array for TRAINING data only
    train_tusimple_size = train_dataset.tusimple_train_size
    train_sea_size = train_dataset.sea_train_size
    weights = np.zeros(train_dataset.train_size)

    # Calculate weights for equal contribution
    total_samples = train_tusimple_size + train_sea_size
    tusimple_weight = 0.6 / (train_tusimple_size / total_samples)
    sea_weight = 0.4 / (train_sea_size / total_samples)

    # Apply weights to all samples
    for i in range(train_dataset.train_size):
        if i < train_tusimple_size:
            weights[i] = tusimple_weight
        else:
            weights[i] = sea_weight

    # Create sampler for TRAINING only
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    print(f"Created weighted sampler: TuSimple weight={tusimple_weight:.4f}, SEA weight={sea_weight:.4f}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True,
        # sampler=sampler,
        num_workers=os.cpu_count() // 2
    )
    
    # val_loader = DataLoader(
    #     val_dataset, 
    #     batch_size=8, 
    #     shuffle=False, 
    #     num_workers=os.cpu_count() // 2
    # )
    
    # Initialize model
    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train model
    model = train_model(model, train_loader, criterion, optimizer, device, epochs=20)

if __name__ == '__main__':
    main()

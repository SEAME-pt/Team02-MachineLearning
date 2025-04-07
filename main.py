import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from src.CombinedDataset import CombinedLaneDataset
from src.SEAMEDataset import SEAMEDataset
from src.train import train_model
from src.unet import UNet
import os

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

    # "/Users/ruipedropires/LaneNet/assets/TUSimple/train_set/label_data_0313.json",
    # "/Users/ruipedropires/LaneNet/assets/TUSimple/train_set/label_data_0601.json"

    # Your dataset configs
    tusimple_config = {
        'json_paths': ["/Users/ruipedropires/LaneNet/assets/TUSimple/train_set/label_data_0531.json"],
        'img_dir': '/Users/ruipedropires/LaneNet/assets/TUSimple/train_set/',
        'width': 256,
        'height': 128,
        'is_train': True,  # Set is_train=True for the base dataset
        'thickness': 3
    }
    
    sea_config = {
        'img_dir': '/Users/ruipedropires/SEAME/Dataset/frames',
        'mask_dir': '/Users/ruipedropires/SEAME/Dataset/masks',
        'width': 256,
        'height': 128,
        'is_train': True  # Set is_train=True for the base dataset
    }
    
    # Create the combined dataset with built-in train/val split
    combined_dataset = CombinedLaneDataset(tusimple_config, sea_config, val_split=0.1)
    
    # Get train and val datasets (these are views of the same dataset with different modes)
    train_dataset = combined_dataset.get_train_dataset()
    val_dataset = combined_dataset.get_val_dataset()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=os.cpu_count() // 2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=os.cpu_count() // 2
    )
    
    # Initialize model
    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50)

if __name__ == '__main__':
    main()

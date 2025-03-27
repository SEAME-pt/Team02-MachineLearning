import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from src.Dataset import TuSimpleDataset
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
    
    # Dataset paths
    json_paths =[
        "/home/luis_t2/LaneNet/assets/TUSimple/train_set/label_data_0313.json",
        "/home/luis_t2/LaneNet/assets/TUSimple/train_set/label_data_0531.json",
        "/home/luis_t2/LaneNet/assets/TUSimple/train_set/label_data_0601.json"
    ]
    img_dir = "/home/luis_t2/LaneNet/assets/TUSimple/train_set/"
    
    # Create dataset with augmentation
    full_dataset = TuSimpleDataset(json_paths, img_dir, width=256, height=128, is_train=True)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Disable augmentation for validation
    val_dataset.dataset.is_train = False
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=os.cpu_count() // 2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=os.cpu_count() // 2)
    
    # Initialize model
    model = UNet().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=15)

if __name__ == '__main__':
    main()

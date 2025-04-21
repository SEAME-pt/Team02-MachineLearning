import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from src.CombinedDataset import CombinedLaneDataset
from src.ObjectDetection import SimpleYOLO, YOLOLoss, generate_anchors
from src.train import train_model, train_yolo_model
from src.unet import UNet
import os
import numpy as np

def yolo_collate_fn(batch):
        """
        Custom collate function for batching tuples with variable-sized object targets
        """
        # Unpack the batch of tuples
        images, masks, targets = zip(*batch)
        
        # Stack images and masks (which have fixed sizes)
        images = torch.stack(images)
        masks = torch.stack(masks)
        
        # Keep targets as a list (each image can have different number of objects)
        # No need to stack these
        
        return images, masks, targets

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

    # Your dataset configs
    tusimple_config = {
        'json_paths': ["/Users/ruipedropires/SEAME/LaneAnchor/assets/TUSimple/train_set/label_data_0313.json",
                      "/Users/ruipedropires/SEAME/LaneAnchor/assets/TUSimple/train_set/label_data_0531.json",
                      "/Users/ruipedropires/SEAME/LaneAnchor/assets/TUSimple/train_set/label_data_0601.json"],
        'img_dir': '/Users/ruipedropires/SEAME/LaneAnchor/assets/TUSimple/train_set/',
        'width': 256,
        'height': 128,
        'is_train': True,
        'thickness': 3
    }

    carla_config = {
        'json_paths': ["/Users/ruipedropires/carla/PythonAPI/Carla-Lane-Detection-Dataset-Generation/data/dataset/Town03_Opt/train_gt.json"],
        'img_dir': '/Users/ruipedropires/carla/PythonAPI/Carla-Lane-Detection-Dataset-Generation/',
        'width': 256,
        'height': 128,
        'is_train': True,
        'thickness': 3
    }
    
    sea_config = {
        'img_dir': '/Users/ruipedropires/SEAME/Dataset/frames',
        'mask_dir': '/Users/ruipedropires/SEAME/Dataset/masks',
        'width': 256,
        'height': 128,
        'is_train': True
    }
    
    bdd100k_config = {
        'img_dir': '/Users/ruipedropires/SEAME/bdd100k/bdd100k/images/100k/train',
        'labels_file': '/Users/ruipedropires/SEAME/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json',
        'width': 256,
        'height': 128,
        'is_train': True,
        'thickness': 3
    }
    
    # Create the combined dataset with built-in train/val split
    combined_dataset = CombinedLaneDataset(
        # tusimple_config=tusimple_config, 
        # sea_config=sea_config, 
        # carla_config=carla_config, 
        bdd100k_config=bdd100k_config,  # Add BDD100K config
        val_split=0.0
    )
    
    # Get train and val datasets
    train_dataset = combined_dataset.get_train_dataset()

    # Create weights array for TRAINING data only
    train_tusimple_size = train_dataset.tusimple_train_size
    train_sea_size = train_dataset.sea_train_size
    train_carla_size = train_dataset.carla_train_size
    train_bdd100k_size = train_dataset.bdd100k_train_size  # Add BDD100K size
    weights = np.zeros(train_dataset.train_size)

    # Calculate weights for equal contribution (adjust percentages as needed)
    total_samples = train_tusimple_size + train_sea_size + train_carla_size + train_bdd100k_size
    tusimple_weight = 0.4 / (train_tusimple_size / total_samples) if train_tusimple_size > 0 else 0
    sea_weight = 0.2 / (train_sea_size / total_samples) if train_sea_size > 0 else 0
    carla_weight = 0.2 / (train_carla_size / total_samples) if train_carla_size > 0 else 0
    bdd100k_weight = 0.2 / (train_bdd100k_size / total_samples) if train_bdd100k_size > 0 else 0

    # Apply weights to all samples
    for i in range(train_dataset.train_size):
        if i < train_tusimple_size:
            weights[i] = tusimple_weight
        elif i < train_tusimple_size + train_sea_size:
            weights[i] = sea_weight
        elif i < train_tusimple_size + train_sea_size + train_carla_size:
            weights[i] = carla_weight
        else:
            weights[i] = bdd100k_weight

    # Create sampler for TRAINING only
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    print(f"Created weighted sampler: TuSimple={tusimple_weight:.4f}, SEA={sea_weight:.4f}, Carla={carla_weight:.4f}, BDD100K={bdd100k_weight:.4f}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        sampler=sampler,
        num_workers=os.cpu_count() // 2,
        collate_fn=yolo_collate_fn
    )
    
    # # Initialize model
    # model = UNet().to(device)
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train model
    # model = train_model(model, train_loader, criterion, optimizer, device, epochs=20)

    # YOLO model and loss parameters
    num_classes = 6  # Assuming 6 classes: car, bus, truck, pedestrian, traffic light, traffic sign
    input_size = 256  # Match your dataset input size
    
    # Generate anchors for your dataset
    # These should be tuned for your specific dataset - these are just example values
    anchors = generate_anchors(num_anchors=9, input_size=input_size)
    anchors = torch.tensor(anchors).float()  # Convert to tensor

    # Initialize YOLO model for object detection
    yolo_model = SimpleYOLO(num_classes=num_classes, anchors=anchors).to(device)
    yolo_optimizer = optim.Adam(yolo_model.parameters(), lr=1e-4)

    yolo_criterion = YOLOLoss(
        anchors=anchors,
        num_classes=num_classes,
        input_dim=input_size,
        device=device
    )

    trained_yolo_model = train_yolo_model(
        yolo_model, 
        train_loader, 
        yolo_criterion, 
        yolo_optimizer, 
        device, 
        epochs=20
    )


if __name__ == '__main__':
    main()
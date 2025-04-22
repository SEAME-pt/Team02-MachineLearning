import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from src.CombinedDataset import CombinedLaneDataset
from src.COCODataset import COCODataset
from src.ObjectDetection import SimpleYOLO, YOLOLoss
from src.anchors import generate_anchors
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

def collate_fn(batch):
    """
    Custom collate function for object detection batches
    with variable number of objects per image
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, 0)
    
    return images, targets

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

    coco_train_dir = '/home/luis_t2/SEAME/train2017'
    coco_ann_file = '/home/luis_t2/SEAME/annotations/instances_train2017.json'

    # Map COCO categories to your custom indices (optional)
    class_map = {
        1: 0,    # person - critical for pedestrian detection
        2: 1,    # bicycle - cyclists on roadways
        3: 2,    # car - primary vehicle type
        4: 3,    # motorcycle - smaller vehicles with different dynamics
        6: 4,    # bus - large vehicles
        8: 5,    # truck - large vehicles with different behavior
        10: 6,   # traffic light - critical for navigation
        13: 7,   # stop sign
        17: 8,   # cat - animals that might cross roads
        18: 9,   # dog - animals that might cross roads
        41: 10,  # skateboard - alternative transportation on roads
        63: 11,  # laptop - might indicate distracted pedestrians
        67: 12,  # cell phone - indicates distracted pedestrians/drivers
        73: 13,  # laptop - might indicate distracted pedestrians
    }

    # Initialize dataset
    coco_dataset = COCODataset(
        img_dir=coco_train_dir,
        annotations_file=coco_ann_file,
        width=384, 
        height=192,  # Same as in your BDD100K setup
        class_map=class_map,
        is_train=True
    )

    train_loader = DataLoader(
        coco_dataset, 
        batch_size=8, 
        # sampler=sampler,
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        collate_fn=collate_fn
    )

    num_classes = max(class_map.values())
    input_size = 192 
    
    # Generate anchors for your dataset
    # These should be tuned for your specific dataset - these are just example values
    anchors = generate_anchors(input_size=(384, 192), method='adaptive')
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

    train_yolo_model(
        yolo_model, 
        train_loader, 
        yolo_criterion, 
        yolo_optimizer, 
        device, 
        epochs=40
    )


if __name__ == '__main__':
    main()
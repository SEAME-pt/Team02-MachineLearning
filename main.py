import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from src.COCODataset import COCODataset
from src.ObjectDetection import SimpleYOLO, YOLOLoss
from src.train import train_yolo_model
import os

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
    }

    num_classes = max(class_map.values()) + 1
    input_size = (384, 192) 

    # Initialize dataset
    coco_dataset = COCODataset(
        img_dir=coco_train_dir,
        annotations_file=coco_ann_file,
        width=input_size[0], 
        height=input_size[1],
        class_map=class_map,
        is_train=True
    )

    train_loader = DataLoader(
        coco_dataset, 
        batch_size=16, 
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        collate_fn=collate_fn
    )

    # Initialize YOLO model for object detection
    yolo_model = SimpleYOLO(
        num_classes=num_classes, 
        input_size=input_size,
        use_default_anchors=False
    ).to(device)

    yolo_criterion = YOLOLoss(
        anchors=yolo_model.anchors,
        num_classes=num_classes,
        input_dim=input_size[1],
        device=device
    )

    yolo_optimizer = optim.Adam(yolo_model.parameters(), lr=1.5e-4)

    train_yolo_model(
        yolo_model, 
        train_loader, 
        yolo_criterion, 
        yolo_optimizer, 
        device, 
        epochs=100
    )


if __name__ == '__main__':
    main()
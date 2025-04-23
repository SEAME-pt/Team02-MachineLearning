import os
import json
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torchvision.transforms as transforms
from .augmentation import ObjectDetectionAugmentation

def get_image_transform():
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t = [transforms.ToTensor(), normalizer]
    return transforms.Compose(t)

class COCODataset(Dataset):
    def __init__(self, img_dir, annotations_file, width=512, height=256, is_train=True, 
                 include_crowd=False, max_objects=50, class_map=None):
        """
        COCO Dataset for object detection
        
        Args:
            img_dir: Directory containing images (e.g., 'coco/train2017/')
            annotations_file: Path to COCO annotation JSON file
            width: Target image width
            height: Target image height
            is_train: Whether this is for training
            include_crowd: Whether to include crowd annotations
            max_objects: Maximum number of objects per image
            class_map: Optional dictionary to map COCO categories to your custom indices
        """
        self.img_dir = img_dir
        self.coco = COCO(annotations_file)
        self.width = width
        self.height = height
        self.is_train = is_train
        self.max_objects = max_objects
        self.transform = get_image_transform()
        if is_train:
            self.augmentation = ObjectDetectionAugmentation(height=height, width=width)
        else:
            self.augmentation = None
        # Get all valid image IDs (those with annotations)
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        
        # Setup category mapping
        self.categories = sorted(self.coco.getCatIds())
        if class_map is None:
            # Create a mapping from COCO ID to index
            self.class_map = {cat_id: i for i, cat_id in enumerate(self.categories)}
        else:
            self.class_map = class_map
            
        self.num_classes = len(self.class_map)
        
        # Create reverse mapping from index to category name
        self.id_to_name = {}
        for cat_id, idx in self.class_map.items():
            cat_info = self.coco.loadCats(cat_id)[0]
            self.id_to_name[idx] = cat_info['name']
        
        # Filter images based on whether we want to include crowd annotations
        if not include_crowd:
            ids = []
            for img_id in self.img_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                annotations = self.coco.loadAnns(ann_ids)
                if all(anno.get('iscrowd', 0) == 0 for anno in annotations):
                    ids.append(img_id)
            self.img_ids = ids
            
        print(f"Loaded {len(self.img_ids)} COCO images with {self.num_classes} classes")
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        # Get image ID and metadata
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original dimensions
        orig_h, orig_w = image.shape[:2]
        
        # Resize image
        image = cv2.resize(image, (self.width, self.height))
        
        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Prepare target tensors for object detection (YOLO format)
        targets = []
        class_labels = []
        
        for anno in annotations:
            # Skip crowd annotations
            if anno.get('iscrowd', 0) == 1:
                continue
            
            # Skip if the category is not in our mapping
            if anno['category_id'] not in self.class_map:
                continue
                
            # Get bbox coordinates
            bbox = anno['bbox']  # [x, y, width, height] in COCO format
            
            # Convert to normalized YOLO format: [class_id, x_center, y_center, width, height]
            x_center = (bbox[0] + bbox[2] / 2) / orig_w
            y_center = (bbox[1] + bbox[3] / 2) / orig_h
            width = bbox[2] / orig_w
            height = bbox[3] / orig_h
            
            # Skip invalid boxes
            if width <= 0 or height <= 0:
                continue
                
            # Get class ID based on our mapping
            class_id = self.class_map[anno['category_id']]
            
            targets.append([x_center, y_center, width, height])
            class_labels.append(class_id)

        # Apply augmentation if training
        if self.is_train and self.augmentation is not None and targets:
            try:
                transformed = self.augmentation(image=image, bboxes=targets, class_labels=class_labels)
                image = transformed['image']  # This is now a tensor
                bboxes = transformed['bboxes']
                class_labels = transformed['class_labels']
                
                # ADD THIS CODE: Format and return successfully augmented data
                if bboxes:
                    targets = []
                    for i in range(len(bboxes)):
                        targets.append([class_labels[i]] + list(bboxes[i]))
                    targets = torch.tensor(targets, dtype=torch.float32)
                else:
                    targets = torch.zeros((0, 5), dtype=torch.float32)
                return image, targets
            except ValueError as e:
                # If augmentation fails, fall back to just the transform
                print(f"Augmentation error on image {img_id}, using basic transform instead")
                if targets:
                    for i in range(len(targets)):
                        targets[i] = [class_labels[i]] + targets[i]
                    targets = torch.tensor(targets, dtype=torch.float32)
                else:
                    targets = torch.zeros((0, 5), dtype=torch.float32)
                return image, targets
        else:
            # No augmentation, just transform the image and keep original targets
            # Convert to tensor if we have targets
            if targets:
                # Add class_id to each box
                for i in range(len(targets)):
                    targets[i] = [class_labels[i]] + targets[i]
                targets = torch.tensor(targets, dtype=torch.float32)
            else:
                targets = torch.zeros((0, 5), dtype=torch.float32)
            
            # Apply transforms
            image = self.transform(image)
            return image, targets
        
    
    def interactive_visualize(self, start_idx=0):
        """
        Interactive visualization using OpenCV that allows navigating through samples
        
        Controls:
        - Right arrow or 'n': next image
        - Left arrow or 'p': previous image
        - 'q' or ESC: quit the visualization
        
        Args:
            start_idx: Index to start visualization from
        """
        # Validate start index
        idx = start_idx if 0 <= start_idx < len(self) else 0
        
        print("==== COCO Dataset Interactive Visualization ====")
        print("Controls:")
        print("  → (Right Arrow) or 'n': Next image")
        print("  ← (Left Arrow) or 'p': Previous image")
        print("  q or ESC: Quit")
        
        while True:
            # Get image ID and metadata
            img_id = self.img_ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            
            # Load image
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            image = cv2.imread(img_path)
            if image is None:
                print(f"Could not read image: {img_path}, skipping...")
                idx = (idx + 1) % len(self)
                continue
            
            # Get original dimensions
            orig_h, orig_w = image.shape[:2]
            
            # Resize for display
            image_display = cv2.resize(image, (self.width, self.height))
            
            # Create a copy for overlays
            overlay = image_display.copy()
            
            # Get processed data
            _, obj_targets = self.__getitem__(idx)
            
            # Convert tensor to numpy if needed
            if isinstance(obj_targets, torch.Tensor):
                obj_targets = obj_targets.detach().cpu().numpy()
            
            # Draw bounding boxes for objects
            for box in obj_targets:
                class_id, x_center, y_center, width, height = box
                
                # Convert normalized coordinates to pixel
                x_center_px = int(x_center * self.width)
                y_center_px = int(y_center * self.height)
                width_px = int(width * self.width)
                height_px = int(height * self.height)
                
                # Calculate top-left and bottom-right corners
                x1 = int(x_center_px - width_px / 2)
                y1 = int(y_center_px - height_px / 2)
                x2 = int(x_center_px + width_px / 2)
                y2 = int(y_center_px + height_px / 2)
                
                # Ensure coordinates are within bounds
                x1 = max(0, min(x1, self.width - 1))
                y1 = max(0, min(y1, self.height - 1))
                x2 = max(0, min(x2, self.width - 1))
                y2 = max(0, min(y2, self.height - 1))
                
                # Get class name
                class_id = int(class_id)
                class_name = self.id_to_name.get(class_id, f"Class {class_id}")
                
                # Generate a unique color for this class (for visual variety)
                color_hash = int(class_id * 5757) % 255  # Simple hash function
                color = (color_hash, 255 - color_hash, 150)
                
                # Draw bounding box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                
                # Add text background for better readability
                text_size, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
                
                # Add class label
                cv2.putText(overlay, class_name, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add text with sample info
            info_text = f"COCO: Sample {idx}/{len(self)-1}: {img_info['file_name']}"
            cv2.putText(overlay, info_text, (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Add object count
            count_text = f"Objects: {len(obj_targets)}"
            cv2.putText(overlay, count_text, (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Add image resolution
            resolution_text = f"Original resolution: {orig_w}x{orig_h}"
            cv2.putText(overlay, resolution_text, (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show the visualization
            cv2.imshow('COCO Dataset Visualization', overlay)
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            
            # Handle key presses
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                break
            elif key == 83 or key == ord('n'):  # Right arrow or 'n' for next
                idx = (idx + 1) % len(self)
            elif key == 81 or key == ord('p'):  # Left arrow or 'p' for previous
                idx = (idx - 1) % len(self)
        
        # Clean up
        cv2.destroyAllWindows()
        print("Visualization ended")

# Simple usage example
if __name__ == "__main__":
    
    # Setup COCO dataset
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
        width=1920, 
        height=1080,  # Same as in your BDD100K setup
        class_map=class_map,
        is_train=True
    )

    if len(coco_dataset) > 0:
        # Start visualization from a random sample
        import random
        random_idx = random.randint(0, len(coco_dataset) - 1)
        coco_dataset.interactive_visualize(start_idx=random_idx)
import os
import json
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

def get_image_transform():
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t = [transforms.ToTensor(), normalizer]
    return transforms.Compose(t)

class BDD100KDataset(Dataset):
    def __init__(self, img_dir, labels_file, width=512, height=256, is_train=True):
        """
        BDD100K Dataset for lane detection and object detection with direct path specification
        
        Args:
            img_dir: Directory containing images (e.g. '/bdd100k/bdd100k/images/100k/train')
            labels_file: Path to labels JSON file containing both object and lane annotations
            width: Target image width
            height: Target image height
            is_train: Whether this is for training (enables augmentations)
        """
        self.img_dir = img_dir
        self.labels_file = labels_file
        self.width = width
        self.height = height
        self.is_train = is_train
        self.transform = get_image_transform()
        
        print(f"Using image directory: {self.img_dir}")
        print(f"Using labels file: {self.labels_file}")
        
        # Load annotations
        with open(self.labels_file, 'r') as f:
            self.annotations = json.load(f)
        print(f"Loaded {len(self.annotations)} annotations")
        
        # Create image name to annotation mapping
        self.img_to_annot = {}
        for item in self.annotations:
            self.img_to_annot[item['name']] = item
        
        # Define object categories we care about
        self.obj_categories = ['car', 'bus', 'truck', 'pedestrian', 'traffic light', 'traffic sign']
        self.category_to_id = {cat: i for i, cat in enumerate(self.obj_categories)}
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        print(f"Found {len(self.image_files)} images")
        
        # Create samples list - images that have both lane and object annotations
        self.samples = []
        valid_images = 0
        
        for img_file in self.image_files:
            if img_file in self.img_to_annot:
                annotation = self.img_to_annot[img_file]
                
                # Check if image has lane annotations
                has_lane = False
                # Check if image has object detection annotations with our categories
                has_obj_detection = False
                
                for label in annotation.get('labels', []):
                    if 'category' in label:
                        if label['category'] == 'lane':
                            has_lane = True
                        elif label['category'] in self.obj_categories:
                            has_obj_detection = True
                    
                    # Break early if we found both
                    if has_lane and has_obj_detection:
                        break
                
                # Only include if we have both lane and object annotations
                if has_lane and has_obj_detection:
                    img_path = os.path.join(self.img_dir, img_file)
                    self.samples.append((img_path, img_file))
                    valid_images += 1
        
        print(f"Found {valid_images} valid images with both lane and object annotations")
        
        if len(self.samples) == 0:
            raise ValueError("No valid images found with both lane and object annotations")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get paths from samples
        img_path, img_name = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Resize image
        resized_img = cv2.resize(image, (self.width, self.height))
        
        # Get annotation for this image
        annotation = self.img_to_annot[img_name]
        
        # Create lane mask
        lane_mask = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Format for object detection (YOLO format)
        # [class_id, x_center, y_center, width, height] (normalized)
        obj_targets = []
        
        # Process all labels
        for label in annotation.get('labels', []):
            if 'category' not in label:
                continue
                
            # Process lane annotations
            if label['category'] == 'lane' and 'poly2d' in label:
                for poly in label['poly2d']:
                    if 'vertices' in poly:
                        vertices = poly['vertices']
                        # Convert vertices to points in resized image
                        points = []
                        for vertex in vertices:
                            x, y = vertex
                            # Scale to resized dimensions
                            x_scaled = int(x * self.width / orig_w)
                            y_scaled = int(y * self.height / orig_h)
                            points.append((x_scaled, y_scaled))
                        
                        # Draw lane line on mask
                        if len(points) > 1:
                            for i in range(len(points) - 1):
                                cv2.line(lane_mask, points[i], points[i+1], 1.0, thickness=3)
            
            # Process object detection annotations
            elif label['category'] in self.obj_categories and 'box2d' in label:
                category = label['category']
                class_id = self.category_to_id[category]
                box = label['box2d']
                
                # Extract coordinates
                x1, y1 = box['x1'], box['y1']
                x2, y2 = box['x2'], box['y2']
                
                # Convert to YOLO format (normalized)
                x_center = ((x1 + x2) / 2) / orig_w
                y_center = ((y1 + y2) / 2) / orig_h
                width = (x2 - x1) / orig_w
                height = (y2 - y1) / orig_h
                
                # Add to targets
                obj_targets.append([class_id, x_center, y_center, width, height])
        
        # Convert lane mask to required format
        lane_mask = lane_mask[None, ...]  # Add channel dimension
        
        # Convert targets to tensor
        obj_targets = torch.tensor(obj_targets, dtype=torch.float32) if obj_targets else torch.zeros((0, 5), dtype=torch.float32)
        
        # Apply transformations
        transformed_img = self.transform(resized_img)
        
        return {
            'image': transformed_img,
            'lane_mask': lane_mask,
            'obj_boxes': obj_targets,
            'image_name': img_name
        }

    def visualize(self, indices=None, num_samples=3):
        """
        Visualize images with their lane masks and object boxes
        
        Args:
            indices: List of specific indices to visualize, or None for random samples
            num_samples: Number of samples to visualize if indices is None
        """
        import matplotlib.pyplot as plt
        import random
        
        # If no indices provided, choose random samples
        if indices is None:
            if len(self) <= num_samples:
                indices = list(range(len(self)))
            else:
                indices = random.sample(range(len(self)), num_samples)
        # If a single index is provided, convert to list
        elif isinstance(indices, int):
            indices = [indices]
        
        # Create figure
        fig = plt.figure(figsize=(15, 5 * len(indices)))
        
        for i, idx in enumerate(indices):
            # Get sample
            sample = self[idx]
            image = sample['image']
            lane_mask = sample['lane_mask']
            obj_boxes = sample['obj_boxes']
            img_name = sample['image_name']
            
            # Convert tensor to numpy for visualization
            image = image.permute(1, 2, 0).numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)
            
            # Create subplots for this sample
            ax1 = fig.add_subplot(len(indices), 2, i*2 + 1)
            ax2 = fig.add_subplot(len(indices), 2, i*2 + 2)
            
            # Show original image with boxes
            ax1.imshow(image)
            ax1.set_title(f"Image {idx}: {img_name}")
            
            # Draw boxes
            for box in obj_boxes:
                class_id, x_center, y_center, width, height = box.tolist()
                # Convert normalized coordinates to pixel values
                x1 = int((x_center - width/2) * self.width)
                y1 = int((y_center - height/2) * self.height)
                x2 = int((x_center + width/2) * self.width)
                y2 = int((y_center + height/2) * self.height)
                
                # Draw rectangle
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
                ax1.add_patch(rect)
                
                # Get category name
                category = self.obj_categories[int(class_id)]
                ax1.text(x1, y1-5, category, color='white', backgroundcolor='red', fontsize=8)
            
            # Show lane mask
            lane_mask = lane_mask[0]  # Remove channel dimension
            ax2.imshow(lane_mask, cmap='gray')
            ax2.set_title(f"Lane Mask {idx}")
        
        plt.tight_layout()
        plt.show()


# Simple usage example with direct paths
if __name__ == "__main__":
    # Direct paths to dataset components - replace with your actual paths
    img_dir = "/home/luis_t2/SEAME/bdd100k/bdd100k/images/100k/train"
    labels_file = "/home/luis_t2/SEAME/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
    
    # Create dataset with direct paths
    dataset = BDD100KDataset(
        img_dir=img_dir,
        labels_file=labels_file
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Visualize a sample if dataset is not empty
    if len(dataset) > 0:
        dataset.visualize(num_samples=5)
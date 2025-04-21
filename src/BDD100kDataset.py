import os
import json
import cv2
import numpy as np
import torch
# from .augmentation import LaneDetectionAugmentation
import torchvision.transforms as transforms
from torch.utils.data import Dataset

def get_image_transform():
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t = [transforms.ToTensor(), normalizer]
    return transforms.Compose(t)

class BDD100KDataset(Dataset):
    def __init__(self, img_dir, labels_file, width=512, height=256, is_train=True, thickness=5):
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
        self.thickness = thickness  # Added thickness parameter
        self.transform = get_image_transform()
        # self.augmentation = LaneDetectionAugmentation(
        #     height=height, 
        #     width=width,
        # )
        
        # Load annotations
        with open(self.labels_file, 'r') as f:
            self.annotations = json.load(f)
        print(f"Loaded {len(self.annotations)} annotations")
        
        # Create image name to annotation mapping
        self.img_to_annot = {}
        for item in self.annotations:
            self.img_to_annot[item['name']] = item
        
        # Define object categories we care about
        self.obj_categories = ['car', 'bus', 'truck', 'pedestrian', 'traffic light']
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
        image = cv2.resize(image, (self.width, self.height))
        
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
        bin_labels = lane_mask.astype(np.float32)[None, ...]  # Add channel dimension
        
        if obj_targets:
            obj_targets = torch.tensor(obj_targets, dtype=torch.float32)
        else:
            obj_targets = torch.zeros((0, 5), dtype=torch.float32)

        # # Apply transformations based on training mode
        # if self.is_train:
        #     return self.augmentation(image, bin_labels)
        # else:
        image = self.transform(image)
        bin_labels = torch.from_numpy(lane_mask.astype(np.float32)[None, ...]) 
        return image, bin_labels, obj_targets

    def interactive_visualize(self, start_idx=0):
        """
        Interactive visualization using OpenCV that allows navigating through samples
        
        Controls:
        - Right arrow: next image
        - Left arrow: previous image
        - 'q' or ESC: quit the visualization
        
        Args:
            start_idx: Index to start visualization from
        """
        import random
        
        idx = start_idx if 0 <= start_idx < len(self) else 0
        
        print("==== BDD100K Interactive Visualization ====")
        print("Controls:")
        print("  → (Right Arrow): Next image")
        print("  ← (Left Arrow): Previous image")
        print("  q or ESC: Quit")
        
        while True:
            # Get sample data
            img_path, img_name = self.samples[idx]
            
            # Load image
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
            _, lane_mask, obj_targets = self.__getitem__(idx)
            
            # Convert tensors if needed
            if isinstance(lane_mask, torch.Tensor):
                lane_mask = lane_mask.detach().cpu().numpy()
            
            if isinstance(obj_targets, torch.Tensor):
                obj_targets = obj_targets.detach().cpu().numpy()
            
            # Create separate visualizations
            if lane_mask.ndim > 2:
                lane_mask = lane_mask[0]  # Remove channel dimension
            
            # Create colored lane mask for overlay
            lane_overlay = np.zeros_like(image_display)
            lane_binary = (lane_mask > 0.5).astype(np.uint8)
            lane_colored = cv2.cvtColor(lane_binary * 255, cv2.COLOR_GRAY2BGR)
            lane_colored[lane_binary > 0] = [0, 255, 255]  # Yellow for lanes
            
            # Blend the lane overlay with the image
            cv2.addWeighted(image_display, 0.7, lane_colored, 0.3, 0, overlay)
            
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
                class_name = self.obj_categories[class_id] if class_id < len(self.obj_categories) else f"Class {class_id}"
                
                # Draw bounding box
                color = (0, 0, 255)  # Red for objects
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                
                # Add text background
                text_size, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
                
                # Add class label
                cv2.putText(overlay, class_name, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add text with sample info
            info_text = f"Sample {idx}/{len(self)-1}: {os.path.basename(img_name)}"
            cv2.putText(overlay, info_text, (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show the combined visualization
            cv2.imshow('BDD100K Dataset Visualization', overlay)
            
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
    # Direct paths to dataset components
    img_dir = "/home/luis_t2/SEAME/bdd100k/bdd100k/images/100k/train"
    labels_file = "/home/luis_t2/SEAME/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
    
    # Create dataset with direct paths
    dataset = BDD100KDataset(
        img_dir=img_dir,
        labels_file=labels_file,
        width=512,
        height=256,
        is_train=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Use interactive visualization to navigate through samples
    if len(dataset) > 0:
        # Start visualization from a random sample
        import random
        random_idx = random.randint(0, len(dataset) - 1)
        dataset.interactive_visualize(start_idx=random_idx)
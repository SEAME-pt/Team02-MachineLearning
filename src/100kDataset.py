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

    t = [transforms.ToTensor(),
         normalizer]

    transform = transforms.Compose(t)
    return transform

class BDD100KDataset(Dataset):
    def __init__(self, root_dir, split='train', width=512, height=256, is_train=True):
        """
        BDD100K Dataset for lane detection and object detection
        
        Args:
            root_dir: Root directory of BDD100K dataset
            split: 'train' or 'val'
            width: Target image width
            height: Target image height
            is_train: Whether this is for training (enables augmentations)
        """
        self.root_dir = root_dir
        self.width = width
        self.height = height
        self.is_train = is_train
        self.transform = get_image_transform()
        
        # Define paths for images, lane annotations and object detection
        self.img_dir = os.path.join(root_dir, 'images', '100k', split)
        self.det_labels_file = os.path.join(root_dir, 'labels', 'det_20', f'det_{split}.json')
        self.lane_dir = os.path.join(root_dir, 'labels', 'lane', split)
        
        # Load object detection annotations
        with open(self.det_labels_file, 'r') as f:
            self.det_annotations = json.load(f)
        
        # Create image name to detection annotation mapping
        self.img_to_det = {}
        for item in self.det_annotations:
            self.img_to_det[item['name']] = item
        
        # Define object categories we care about
        self.obj_categories = ['car', 'bus', 'truck', 'pedestrian', 'traffic light', 'traffic sign']
        self.category_to_id = {cat: i for i, cat in enumerate(self.obj_categories)}
        
        # Create samples list - images that have both lane and object annotations
        self.samples = []
        valid_images = 0
        skipped_images = 0
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        
        for img_file in self.image_files:
            # Check if this image has lane annotation
            lane_json_file = img_file.replace('.jpg', '.json')
            lane_json_path = os.path.join(self.lane_dir, lane_json_file)
            
            # Check if image has detection annotations with our categories
            has_obj_detection = False
            if img_file in self.img_to_det:
                det_annot = self.img_to_det[img_file]
                for label in det_annot.get('labels', []):
                    if 'category' in label and label['category'] in self.obj_categories:
                        has_obj_detection = True
                        break
            
            # Only include if we have both lane and object annotations
            if os.path.exists(lane_json_path) and has_obj_detection:
                img_path = os.path.join(self.img_dir, img_file)
                self.samples.append((img_path, lane_json_path, img_file))
                valid_images += 1
            else:
                skipped_images += 1
        
        print(f"BDD100K Dataset loaded: {valid_images} valid images with both lanes and objects, {skipped_images} images skipped")
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid images found with both lane and object annotations. Check your BDD100K directory: {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get paths from samples
        img_path, lane_json_path, img_name = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Resize image
        resized_img = cv2.resize(image, (self.width, self.height))
        
        # Load lane annotation and create lane mask
        with open(lane_json_path, 'r') as f:
            lane_data = json.load(f)
        
        lane_mask = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Parse lane data and draw lanes on mask
        if 'lanes' in lane_data:
            for lane in lane_data['lanes']:
                points = []
                for i in range(len(lane)):
                    if lane[i] >= 0:  # Valid point
                        # Convert to resized image coordinates
                        x = int(i * self.width / len(lane))
                        y = int(lane[i] * self.height / orig_h)
                        points.append((x, y))
                
                # Draw lane line on mask if we have points
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        cv2.line(lane_mask, points[i], points[i+1], 1.0, thickness=5)
        
        # Convert lane mask to required format
        lane_mask = lane_mask[None, ...]  # Add channel dimension
        
        # Get object detection annotations
        obj_annot = self.img_to_det.get(img_name, {'labels': []})
        
        # Format for object detection (YOLO format)
        # [class_id, x_center, y_center, width, height] (normalized)
        obj_targets = []
        
        for label in obj_annot.get('labels', []):
            if 'category' not in label or 'box2d' not in label:
                continue
                
            category = label['category']
            if category not in self.category_to_id:
                continue
                
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
        
        # Convert targets to tensor
        obj_targets = torch.tensor(obj_targets, dtype=torch.float32) if obj_targets else torch.zeros((0, 5), dtype=torch.float32)
        
        # Apply transformations
        if self.is_train:
            # Add augmentation here if needed
            transformed_img = self.transform(resized_img)
        else:
            transformed_img = self.transform(resized_img)
        
        return {
            'image': transformed_img,
            'lane_mask': lane_mask,
            'obj_boxes': obj_targets,
            'image_name': img_name
        }

    def visualize(self, idx):
        """
        Visualize an image with its lane mask and object boxes
        """
        import matplotlib.pyplot as plt
        
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
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Show original image with boxes
        ax1.imshow(image)
        ax1.set_title(f"Image: {img_name}")
        
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
            if int(class_id) < len(self.obj_categories):
                category = self.obj_categories[int(class_id)]
                ax1.text(x1, y1-5, category, color='white', backgroundcolor='red', fontsize=8)
        
        # Show lane mask
        lane_mask = lane_mask[0]  # Remove channel dimension
        ax2.imshow(lane_mask, cmap='gray')
        ax2.set_title("Lane Mask")
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your BDD100K dataset root directory
    root_dir = "/path/to/bdd100k"
    
    # Create dataset
    dataset = BDD100KDataset(root_dir, split='train')
    
    print(f"Dataset size: {len(dataset)}")
    
    # Visualize a sample
    if len(dataset) > 0:
        dataset.visualize(0)
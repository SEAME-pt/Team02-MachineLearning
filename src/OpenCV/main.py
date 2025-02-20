import cv2
import numpy as np
import torch
import glob
from Model import LaneSegmentationModel
from Dataset import get_image_transform
from utils import visualize_output_batch

# Initialize model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model = LaneSegmentationModel().to(device)
model.load_state_dict(torch.load('lane_segmentation.pth', map_location=device))
model.eval()

image_dir = "assets/images"  # Update this to your images directory
image_paths = glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png")

for img_path in image_paths:
    # Read image
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Could not read image: {img_path}")
        continue
        
    height, width, _ = frame.shape
    
    image = cv2.resize(frame, (256, 512))
    transform = get_image_transform()
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)

        visualize_output_batch(image, outputs)

cv2.destroyAllWindows()
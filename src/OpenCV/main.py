import cv2
import numpy as np
import torch
import glob
from Model import LaneSegmentationModel
from Dataset import get_image_transform
from utils import visualize_output_batch
from PIL import Image
from torchvision import transforms

# Initialize model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model = LaneSegmentationModel().to(device)
model.load_state_dict(torch.load('lane_segmentation.pth', map_location=device))
model.eval()

cap = cv2.VideoCapture("assets/road.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    transforms = get_image_transform()
    input_tensor = transforms(rgb_frame).unsqueeze(0).to(device)
    
    # Run inference.
    with torch.no_grad():
        output = model(input_tensor)
    
    output_mask = output.squeeze().cpu().numpy()
    binary_mask = (output_mask > 0.5).astype(np.uint8) * 255
    
    # Resize the mask to the original frame dimensions.
    mask_resized = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]))
    
    # Create a copy of the original frame.
    blended = frame.copy()
    # Replace pixels where the mask is non-zero with green (BGR: [0,255,0]).
    blended[mask_resized > 0] = [0, 255, 0]
    
    # Display the result.
    cv2.imshow("Lane Detection", blended)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
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
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # Optimize CUDA performance
    print(f"Using CUDA device: {torch.cuda.get_device_name()}")
elif torch.backends.mps.is_available():  # For Apple Silicon
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    print("Using CPU")
model = LaneSegmentationModel().to(device)
checkpoint = torch.load("best_lane_segmentation.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

cap = cv2.VideoCapture("assets/road3.mp4")

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
    binary_mask = (output_mask > 0.6).astype(np.uint8) * 255

    kernel = np.ones((4,4), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)  # Remove noise
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel) 
    
    # Resize the mask to the original frame dimensions.
    mask_resized = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]))
    
    # Create a copy of the original frame.
    overlay = frame.copy()
    overlay[mask_resized > 0] = [0, 255, 0]
    alpha = 1  # Transparency factor
    blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Display the result.
    cv2.imshow("Lane Detection", blended)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
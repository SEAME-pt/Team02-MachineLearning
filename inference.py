import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from torchvision import transforms
from src.unet import UNet
import time

# Set up device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name()}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load the trained model
model = UNet().to(device)
model.load_state_dict(torch.load('Models/lane_model5_epoch_13.pth', map_location=device))
model.eval()

# Image preprocessing function
def preprocess_image(image, target_size=(256, 128)):
    # Resize image
    img = cv2.resize(image, target_size)
    
    # 2. Enhance contrast within the ROI
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    
    # Merge channels
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 3. Increase saturation to make lane markings more prominent
    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Increase saturation
    s = cv2.add(s, 30)  # Add 30 to saturation (make colors more vibrant)
    
    # Merge channels
    enhanced_hsv = cv2.merge((h, s, v))
    enhanced_img = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    
    # Convert to RGB for model
    enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
    
    # Apply same transforms as during training
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalizer
    ])
    
    # Apply transforms
    img_tensor = transform(enhanced_rgb).unsqueeze(0).to(device)
    
    return img_tensor, enhanced_img
# Function to overlay lane predictions on image
def overlay_predictions(image, prediction, threshold=0.5):
    # Convert prediction to binary mask
    prediction = prediction.squeeze().cpu().detach().numpy()
    lane_mask = (prediction > threshold).astype(np.uint8) * 255
    
    # Resize mask to match the original image size
    lane_mask = cv2.resize(lane_mask, (image.shape[1], image.shape[0]))
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)  # Remove noise
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel) # Fill small holes
    
    # Connected component analysis to filter small artifacts
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(lane_mask, connectivity=8)
    
    # Filter components based on area
    filtered_mask = np.zeros_like(lane_mask)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= 100:  # Minimum area threshold
            filtered_mask[labels == i] = 255
    
    # Create a colored overlay
    colored_mask = np.zeros_like(image)
    colored_mask[filtered_mask > 0] = [0, 255, 0]  # Green for lane markings
    
    # Apply the overlay with transparency
    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    return overlay

# Open video
cap = cv2.VideoCapture("assets/seame_data.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    time.sleep(1/45) 
    
    # Preprocess the image
    img_tensor, original_frame = preprocess_image(frame)
    
    # Run inference
    with torch.no_grad():
        predictions = model(img_tensor)
        predictions = torch.sigmoid(predictions)
    
    # Overlay predictions on the original frame
    result_frame = overlay_predictions(frame, predictions)
    
    # Display the result
    cv2.imshow("Lane Detection", result_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

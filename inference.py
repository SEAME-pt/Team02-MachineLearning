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
model.load_state_dict(torch.load('Models/lane/lane_UNet1_epoch_2.pth', map_location=device))
model.eval()

# Image preprocessing function
def preprocess_image(image, target_size=(256, 128)):
    # Resize image
    img = cv2.resize(image, target_size)
    
    # 2. Enhance contrast within the ROI
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply same transforms as during training
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalizer
    ])
    
    # Apply transforms
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    return img_tensor, img
# Function to overlay lane predictions on image
def overlay_predictions(image, prediction, threshold=0.6):
    # Convert prediction to binary mask
    prediction = prediction.squeeze().cpu().detach().numpy()
    lane_mask = (prediction > threshold).astype(np.uint8) * 255
    
    # Resize mask to match the original image size
    lane_mask = cv2.resize(lane_mask, (image.shape[1], image.shape[0]))
    
    # Create a colored overlay
    colored_mask = np.zeros_like(image)
    colored_mask[lane_mask > 0] = [0, 255, 0]  # Green for lane markings
    
    # Apply the overlay with transparency
    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    return overlay

# Open video
cap = cv2.VideoCapture("assets/seame_data.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    time.sleep(0.05)  # Optional: Add a small delay to control frame rate
    
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

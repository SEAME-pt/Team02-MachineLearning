import torch
import numpy as np
import cv2
from torchvision import transforms
import time

from src.ObjectDetection import SimpleYOLO, generate_anchors
from src.unet import UNet  # Keep the lane detection model

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

# Class names for visualization
CLASS_NAMES = ['Car', 'Bus', 'Truck', 'Pedestrian', 'Traffic Light', 'Traffic Sign']
# Different colors for each class (BGR format for OpenCV)
COLORS = [
    (0, 0, 255),    # Red for cars
    (0, 165, 255),  # Orange for buses
    (0, 255, 255),  # Yellow for trucks
    (0, 255, 0),    # Green for pedestrians
    (255, 0, 0),    # Blue for traffic lights
    (255, 0, 255)   # Purple for traffic signs
]

# Load the YOLO model
def load_yolo_model(model_path, num_classes=6, input_size=256):
    # Generate anchors
    anchors = generate_anchors(input_size=input_size)
    
    # Create model
    model = SimpleYOLO(num_classes, anchors).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

# Load the lane detection model
def load_lane_model(model_path):
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

# Image preprocessing function (same as in original inference.py)
def preprocess_image(image, target_size=(256, 128)):
    # Resize image
    img = cv2.resize(image, target_size)
    
    # Convert to RGB
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

# Function to overlay lane predictions on image (same as in original inference.py)
def overlay_lane_predictions(image, prediction, threshold=0.6):
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

# Function to draw detected objects on the image
def draw_detections(image, detections, conf_threshold=0.25):
    """
    Draw bounding boxes and labels for detected objects
    
    Args:
        image: Original image to draw on
        detections: List of detection tensors from YOLO model
        conf_threshold: Confidence threshold for displaying detections
    
    Returns:
        Image with bounding boxes and labels drawn
    """
    result_img = image.copy()
    h, w = image.shape[:2]
    
    for detection in detections:
        # Skip if no detections
        if detection.size(0) == 0:
            continue
        
        # Process each detection
        for i in range(detection.size(0)):
            x1, y1, x2, y2, obj_conf, cls_conf, cls_idx = detection[i]
            
            # Skip low confidence detections
            score = float(obj_conf * cls_conf)
            if score < conf_threshold:
                continue
            
            # Convert normalized coordinates to pixel values
            x1 = int(x1.item() * w)
            y1 = int(y1.item() * h)
            x2 = int(x2.item() * w)
            y2 = int(y2.item() * h)
            
            # Get class index and name
            cls_idx = int(cls_idx.item())
            cls_name = CLASS_NAMES[cls_idx]
            
            # Get color for this class
            color = COLORS[cls_idx]
            
            # Draw bounding box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text with confidence
            label = f"{cls_name}: {score:.2f}"
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background
            cv2.rectangle(result_img, 
                         (x1, y1 - text_height - baseline - 5), 
                         (x1 + text_width, y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(result_img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result_img

def visualize_raw_predictions(frame, predictions):
    """Show raw prediction heatmaps for debugging"""
    result = frame.copy()
    
    # Get the first prediction (first batch, first scale)
    pred = predictions[0][0]  # Shape: [3, H, W, C]
    
    # Get objectness confidence from the first anchor
    obj_conf = pred[0, :, :, 4].cpu().numpy()
    
    # Normalize to 0-255 for visualization
    obj_conf = (obj_conf * 255).astype(np.uint8)
    obj_conf = cv2.resize(obj_conf, (frame.shape[1], frame.shape[0]))
    
    # Create a heatmap
    heatmap = cv2.applyColorMap(obj_conf, cv2.COLORMAP_JET)
    
    # Blend with original image
    result = cv2.addWeighted(result, 0.7, heatmap, 0.3, 0)
    
    return result

def main():
    # Load both models
    yolo_model = load_yolo_model('Models/yolo_model_epoch_5.pth')
    lane_model = load_lane_model('Models/temp/lane_model2_epoch_18.pth')
    
    # Set input dimensions
    input_size = (256, 128)  # (width, height)
    
    # Choose video source
    video_path = "assets/road3.mp4"
    cap = cv2.VideoCapture(video_path)

    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Optional: Add a small delay for display
        time.sleep(0.05)
        
        # Preprocess the image for both models
        img_tensor, _ = preprocess_image(frame, target_size=input_size)
        
        # Run inference with both models
        with torch.no_grad():
            # Lane detection
            # lane_predictions = lane_model(img_tensor)
            # lane_predictions = torch.sigmoid(lane_predictions)
            
            # Object detection
            yolo_predictions = yolo_model(img_tensor)
            
            # Post-process object detections
            detections = yolo_model.predict_boxes(
                yolo_predictions, 
                input_dim=input_size[1],  # Height 
                conf_thresh=0.1
            )
            
            # Apply non-maximum suppression to remove overlapping boxes
            processed_detections = []
            for batch_boxes in detections:
                processed_detections.append(
                    yolo_model.non_max_suppression(batch_boxes, nms_thresh=0.45)
                )
        
        # Overlay lane predictions first
        # result_frame = overlay_lane_predictions(frame, lane_predictions)
        
        # Then draw object detections
        # result_frame = draw_detections(frame, processed_detections)
        result_frame = visualize_raw_predictions(frame, yolo_predictions)
        
        # Display processing stats on the frame
        cv2.putText(result_frame, f"Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
        
        # Display the result
        cv2.imshow("Detection Results", result_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    # out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
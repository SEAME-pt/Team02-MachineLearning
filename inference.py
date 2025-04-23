import torch
import numpy as np
import cv2
from torchvision import transforms
import time
from src.ObjectDetection import SimpleYOLO

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

CLASS_NAMES = [
    'Person', 'Bicycle', 'Car', 'Motorcycle', 'Bus', 'Truck',  
    'Traffic Light', 'Stop Sign', 'Cat', 'Dog', 'Skateboard'
]

# Also update COLORS to have enough colors for all classes
COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 255),  # Yellow
    (0, 165, 255),  # Orange
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (255, 0, 255),  # Purple
    (255, 255, 0),  # Cyan
    (128, 0, 255),  # Magenta
    (0, 128, 255),  # Amber
    (255, 0, 128),  # Pink
    (128, 128, 0),  # Olive
]

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

# Function to draw detected objects on the image
def draw_detections(image, detections, conf_threshold=0.05, model_size=(256, 128)):
    result_img = image.copy()
    orig_h, orig_w = image.shape[:2]
    model_w, model_h = model_size
    
    # Calculate scaling factors
    scale_x = orig_w / model_w
    scale_y = orig_h / model_h
    
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
            
            # Scale coordinates to original image dimensions
            x1 = int(x1.item() * scale_x)
            y1 = int(y1.item() * scale_y)
            x2 = int(x2.item() * scale_x)
            y2 = int(y2.item() * scale_y)
            
            # Validate box dimensions
            if x2 <= x1 or y2 <= y1 or x1 >= orig_w or y1 >= orig_h or x2 <= 0 or y2 <= 0:
                continue
            
            # Get the class index and select color
            cls_idx = int(cls_idx.item())
            color = COLORS[cls_idx % len(COLORS)]
            label = f"{CLASS_NAMES[cls_idx]}: {score:.2f}"
            
            # Draw rectangle and label
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text with confidence
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background for better visibility
            cv2.rectangle(result_img, 
                         (x1, y1 - text_height - baseline - 5), 
                         (x1 + text_width, y1), 
                         color, -1)
                         
            cv2.putText(result_img, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result_img

# Improved function to visualize raw predictions with better filtering
def visualize_raw_predictions(frame, predictions, threshold=0.3):
    """
    Show raw prediction heatmaps with thresholding to reduce noise
    
    Args:
        frame: Original image frame
        predictions: Raw YOLO predictions
        threshold: Confidence threshold to filter noise
    """
    result = frame.copy()
    
    # Create a combined heatmap across all scales and anchors
    combined_heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    
    # Process each scale
    for scale_idx, scale_pred in enumerate(predictions):
        pred = scale_pred[0]  # First batch
        
        # Process each anchor
        for anchor_idx in range(pred.size(0)):
            # Get objectness confidence
            obj_conf = pred[anchor_idx, :, :, 4].cpu().numpy()
            
            # Resize to frame dimensions
            resized_conf = cv2.resize(obj_conf, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # Add to combined heatmap with threshold to reduce noise
            combined_heatmap = np.maximum(combined_heatmap, resized_conf * (resized_conf > threshold))
    
    # Normalize the heatmap for visualization
    if np.max(combined_heatmap) > 0:
        combined_heatmap = (combined_heatmap / np.max(combined_heatmap) * 255).astype(np.uint8)
    else:
        combined_heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    
    # Apply a colormap
    heatmap = cv2.applyColorMap(combined_heatmap, cv2.COLORMAP_JET)
    
    # Blend with original image
    result = cv2.addWeighted(result, 0.7, heatmap, 0.3, 0)
    
    # Add helper text
    cv2.putText(result, f"Confidence threshold: {threshold:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return result

def main():
    num_classes = len(CLASS_NAMES) 
    input_size = (384, 192)

    yolo_model = SimpleYOLO(
        num_classes=num_classes, 
        input_size=input_size,
        use_default_anchors=False
    ).to(device)

    yolo_model.load_state_dict(torch.load("Models/Obj/yolo4_model_epoch_1.pth", map_location=device))
    yolo_model.eval()
    
    # Choose video source
    video_path = "assets/road2.mp4"
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
            
            # Object detection
            yolo_predictions = yolo_model(img_tensor)
            
            # Post-process object detections
            detections = yolo_model.predict_boxes(
                yolo_predictions, 
                input_dim=input_size[1],  # Height 
                conf_thresh=0.5
            )
            
            # Apply non-maximum suppression to remove overlapping boxes
            processed_detections = []
            for batch_boxes in detections:
                processed_detections.append(
                    yolo_model.non_max_suppression(batch_boxes, nms_thresh=0.005)
                )
        
        # Then draw object detections
        result_frame = draw_detections(frame, processed_detections, 
                              conf_threshold=0.5, 
                              model_size=input_size)
        # result_frame = visualize_raw_predictions(frame, yolo_predictions)
        
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
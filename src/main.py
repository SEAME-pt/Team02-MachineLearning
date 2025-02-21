import cv2
import numpy as np
import torch
import glob
from Model import LaneSegmentationModel
from Dataset import get_image_transform
from utils import visualize_output_batch
from PIL import Image
from torchvision import transforms


class PostProcessor(object):

    def __init__(self):
        pass

    def process(self, image, kernel_size=5, minarea_threshold=10):
        """Do the post processing here. First the image is converte to grayscale.
        Then a closing operation is applied to fill empty gaps among surrounding
        pixels. After that connected component are detected where small components
        will be removed.

        Args:
            image:
            kernel_size
            minarea_threshold

        Returns:
            image: binary image

        """
        if image.dtype is not np.uint8:
            image = np.array(image, np.uint8)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # fill the pixel gap using Closing operator (dilation followed by
        # erosion)
        kernel = cv2.getStructuringElement(
            shape=cv2.MORPH_RECT, ksize=(
                kernel_size, kernel_size))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        ccs = cv2.connectedComponentsWithStats(
            image, connectivity=8, ltype=cv2.CV_32S)
        labels = ccs[1]
        stats = ccs[2]

        for index, stat in enumerate(stats):
            if stat[4] <= minarea_threshold:
                idx = np.where(labels == index)
                image[idx] = 0

        return image


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

cap = cv2.VideoCapture("assets/road1.mp4")
post_processor = PostProcessor()

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

    processed_mask = post_processor.process(
        binary_mask,
        kernel_size=5,
        minarea_threshold=10
    )
    
    # Resize the mask to the original frame dimensions.
    mask_resized = cv2.resize(processed_mask, (frame.shape[1], frame.shape[0]))
    
    # Create a copy of the original frame.
    overlay = frame.copy()
    overlay[mask_resized > 0] = [0, 255, 0]
    alpha = 0.4  # Transparency factor
    blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Display the result.
    cv2.imshow("Lane Detection", blended)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
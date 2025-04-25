import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from torchvision import transforms
from src.unet import UNet, MobileNetV2UNet
import torch.nn.functional as F
import time
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN

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

input_size = (384, 192)

# Load the trained model
model = MobileNetV2UNet().to(device)
model.load_state_dict(torch.load('Models/lane/lane_mobilenetv2_ins_bin_epoch_29.pth', map_location=device))
model.eval()

# Image preprocessing function
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

def postprocess(image, kernel_size=5, minarea_threshold=50):
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


def cluster(embeddings, bandwidth=1.5):
        """Clustering pixel embedding into lanes using MeanShift

        Args:
            prediction: set of pixel embeddings
            bandwidth: bandwidth used in the RBF kernel

        Returns:
            num_clusters: number of clusters (or lanes)
            labels: lane labels for each pixel
            cluster_centers: centroids

        """
        ms = MeanShift(bandwidth=bandwidth)
        try:
            ms.fit(embeddings)
        except ValueError as err:
            return 0, [], []

        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        num_clusters = cluster_centers.shape[0]

        return num_clusters, labels, cluster_centers


def get_lane_area(binary_seg_ret, instance_seg_ret):
    """ Get possible lane area from the binary segmentation results

    Args:
        binary_seg_ret: Binary segmentation mask
        instance_seg_ret: Instance embedding features

    Returns:
        lane_embedding_feats: Feature embeddings for lane pixels
        lane_coordinate: Coordinates of lane pixels
    """
    # FIX: Check that binary mask is correctly formatted as uint8
    if binary_seg_ret.dtype != np.uint8:
        binary_seg_ret = binary_seg_ret.astype(np.uint8)
    
    # FIX: Find where mask is non-zero instead of exactly 1
    idx = np.where(binary_seg_ret > 0)
    
    # Debug print to see how many pixels were found
    # print(f"Found {len(idx[0])} lane pixels in binary mask")
    
    # Safety check - if no pixels found, return empty arrays
    if len(idx[0]) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
    
    # Get instance embedding features for each lane pixel
    lane_embedding_feats = []
    lane_coordinate = []
    
    # Get correct instance embedding shape
    embed_dim = instance_seg_ret.shape[0] if instance_seg_ret.ndim == 3 else instance_seg_ret.shape[1]
    
    for i in range(len(idx[0])):
        # Check bounds to avoid index errors
        if idx[0][i] < instance_seg_ret.shape[1 if instance_seg_ret.ndim == 3 else 2] and \
           idx[1][i] < instance_seg_ret.shape[2 if instance_seg_ret.ndim == 3 else 3]:
            if instance_seg_ret.ndim == 3:  # [C, H, W]
                lane_embedding_feats.append(instance_seg_ret[:, idx[0][i], idx[1][i]])
            else:  # [B, C, H, W]
                lane_embedding_feats.append(instance_seg_ret[0, :, idx[0][i], idx[1][i]])
            
            lane_coordinate.append([idx[0][i], idx[1][i]])

    # Convert to numpy arrays with proper dtypes
    return np.array(lane_embedding_feats, np.float32), np.array(lane_coordinate, np.int64)


def get_lane_mask(num_clusters, labels, binary_seg_ret, lane_coordinate):
    """
    Get a masking images, where each lane is colored by a different color

    Args:
        num_clusters: number of possible lanes
        labels: lane label for each point
        binary_seg_ret:
        lane_coordinate

    Returns:
        a mask image

    """

    color_map = [(255, 0, 0),
                 (0, 255, 0),
                 (0, 0, 255),
                 (125, 125, 0),
                 (0, 125, 125),
                 (125, 0, 125),
                 (50, 100, 50),
                 (100, 50, 100)]

    # continue working on this
    if num_clusters > 8:
        cluster_sample_nums = []
        for i in range(num_clusters):
            cluster_sample_nums.append(len(np.where(labels == i)[0]))
        sort_idx = np.argsort(-np.array(cluster_sample_nums, np.int64))
        cluster_index = np.array(range(num_clusters))[sort_idx[0:8]]
    else:
        cluster_index = range(num_clusters)

    mask_image = np.zeros(
        shape=[
            binary_seg_ret.shape[0],
            binary_seg_ret.shape[1],
            3],
        dtype=np.uint8)

    for index, ci in enumerate(cluster_index):
        idx = np.where(labels == ci)
        coord = lane_coordinate[idx]
        coord = np.flip(coord, axis=1)
        color = color_map[index]
        coord = np.array([coord])
        cv2.polylines(
            img=mask_image,
            pts=coord,
            isClosed=False,
            color=color,
            thickness=2)

    return mask_image

def overlay_predictions(image, prediction, threshold=0.3):
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
cap = cv2.VideoCapture("assets/road3.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    time.sleep(0.05)  # Optional: Add a small delay to control frame rate
    
    # Preprocess the image
    img_tensor, original_frame = preprocess_image(frame, target_size=input_size)
    
    # Run inference
    with torch.no_grad():
        bin_preds, ins_preds = model(img_tensor)
        bin_preds = torch.sigmoid(bin_preds)
    
    bin_pred = bin_preds.data.cpu().numpy()  
    ins_img = ins_preds.data.cpu().numpy()
    
    lane_prob = bin_pred[0, 0]
    bin_img_raw = (lane_prob > 0.2).astype(np.uint8)

    # Apply post-processing to clean up the mask
    bin_img = postprocess(bin_img_raw, kernel_size=7, minarea_threshold=30)

    overlay_img = frame.copy()
    bin_viz = np.zeros_like(frame)
    bin_resized = cv2.resize(bin_img * 255, (frame.shape[1], frame.shape[0]), 
                           interpolation=cv2.INTER_NEAREST)
    bin_viz[:,:,1] = bin_resized 
    overlay_img = cv2.addWeighted(overlay_img, 0.7, bin_viz, 0.3, 0)
    cv2.imshow("Binary Lane Mask", overlay_img)


    lane_embedding_feats, lane_coordinate = get_lane_area(
                    bin_img, ins_img)
    
    if lane_embedding_feats.size > 0:
        num_clusters, labels, cluster_centers = cluster(lane_embedding_feats, bandwidth=1.5)
        mask_img = get_lane_mask(num_clusters, labels, bin_img, lane_coordinate)
        mask_img = cv2.resize(mask_img, (frame.shape[1], frame.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
        mask_img = mask_img[:, :, (2, 1, 0)]
        overlay_img = cv2.addWeighted(frame, 1.0, mask_img, 1.0, 0)
        cv2.imshow("Lane Detection", overlay_img)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
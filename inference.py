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

input_size = (256, 128)

# Load the trained model
model = UNet().to(device)
model.load_state_dict(torch.load('Models/lane/lane_unet5_ins_ce_epoch_100.pth', map_location=device))
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

def postprocess(image, kernel_size=5, minarea_threshold=20):
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
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
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
        binary_seg_ret:
        instance_seg_ret:

    Returns:

    """
    idx = np.where(binary_seg_ret == 1)

    lane_embedding_feats = []
    lane_coordinate = []
    for i in range(len(idx[0])):
        lane_embedding_feats.append(instance_seg_ret[:, idx[0][i], idx[1][i]])
        lane_coordinate.append([idx[0][i], idx[1][i]])

    return np.array(
        lane_embedding_feats, np.float32), np.array(
        lane_coordinate, np.int64)


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

    # NEW CODE: Order lanes by x-position (left to right)
    ordered_clusters = []
    for ci in cluster_index:
        idx = np.where(labels == ci)
        if len(idx[0]) == 0:
            continue
        
        # Get coordinates for this cluster
        coord = lane_coordinate[idx]
        
        # Calculate average x position (using the bottom third of the lane)
        bottom_y_threshold = binary_seg_ret.shape[0] * 0.7  # Bottom 30% of image
        bottom_points = [pt[1] for pt in coord if pt[0] > bottom_y_threshold]
        
        if bottom_points:
            avg_x = sum(bottom_points) / len(bottom_points)
            ordered_clusters.append((ci, avg_x))
    
    # Sort clusters by x-position (left to right)
    ordered_clusters.sort(key=lambda x: x[1])
    
    # Get sorted cluster indices
    if ordered_clusters:
        cluster_index = [cluster[0] for cluster in ordered_clusters]

    # Rest of your existing code
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

# Open video
cap = cv2.VideoCapture("assets/seame_data.mp4")

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
        bin_preds = F.softmax(bin_preds, dim=1)
    
        bin_pred = bin_preds[0].data.cpu().numpy()  
        ins_img = ins_preds[0].data.cpu().numpy()
        bin_img = bin_pred.argmax(0)

        lane_prob = bin_pred[1]
        bin_img_raw = (lane_prob > 0.5).astype(np.uint8)

        bin_img = postprocess(bin_img_raw, kernel_size=5, minarea_threshold=20)

        lane_embedding_feats, lane_coordinate = get_lane_area(
                        bin_img, ins_img)
        
        if lane_embedding_feats.size > 0:
            num_clusters, labels, cluster_centers = cluster(lane_embedding_feats, bandwidth=1.5)
            mask_img = get_lane_mask(num_clusters, labels, bin_img, lane_coordinate)
            mask_img = cv2.resize(mask_img, (frame.shape[1], frame.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            mask_img = mask_img[:, :, (2, 1, 0)]
        else:
            mask_img = np.zeros((bin_img.shape[0], bin_img.shape[1], 3), np.uint8)
            mask_img = cv2.resize(mask_img, (frame.shape[1], frame.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)

        overlay_img = cv2.addWeighted(frame, 1.0, mask_img, 1.0, 0)
        cv2.imshow("Lane Detection", overlay_img)

        # # Apply post-processing to clean up the mask
        # res = postprocess(bin_img_raw, kernel_size=7, minarea_threshold=30)

        # bin_viz = np.zeros_like(frame)
        # print(frame[0])
        # bin_resized = cv2.resize(res * 255, (frame.shape[1], frame.shape[0]), 
        #                     interpolation=cv2.INTER_NEAREST)
        # bin_viz[:,:,1] = bin_resized  # Show in green channel
        # overlay_img = cv2.addWeighted(frame, 1.0, bin_viz, 1.0, 0)
        # cv2.imshow("Binary Lane Mask", overlay_img)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
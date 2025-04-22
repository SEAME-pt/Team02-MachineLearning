import numpy as np

def generate_optimized_anchors(num_anchors=9, dataset=None):
    """
    Generate anchor boxes using k-means clustering on actual dataset bounding boxes
    
    Args:
        num_anchors: Number of anchor boxes to generate
        dataset: The dataset containing bounding box information
        
    Returns:
        Array of anchor boxes optimized for the dataset, in format [w, h]
    """
    
    # Collect all bounding boxes from dataset
    all_boxes = []
    for i in range(len(dataset)):
        _, targets = dataset[i]
        for box in targets:
            # Extract width and height (normalized)
            w, h = box[3], box[4]
            # Skip any zero-size boxes
            if w > 0 and h > 0:
                all_boxes.append((w, h))
    
    # Convert to numpy array
    all_boxes = np.array(all_boxes)
    
    # Perform k-means clustering
    centroids, _ = kmeans(all_boxes, num_anchors)
    
    # Sort by area
    centroids = centroids[np.argsort(centroids[:, 0] * centroids[:, 1])]
    
    # Denormalize to match input dimensions (256×128)
    centroids[:, 0] *= 256  # width
    centroids[:, 1] *= 128  # height
    
    # Group by scale for multi-scale detection (3 scales with 3 anchors each)
    anchors = centroids.reshape(3, 3, 2)
    
    return anchors

def kmeans(boxes, k):
    """
    Perform k-means clustering using IoU as distance metric
    """
    num_boxes = boxes.shape[0]
    distances = np.empty((num_boxes, k))
    last_clusters = np.zeros((num_boxes,))
    
    # Initialize centroids randomly
    np.random.seed(42)
    clusters = boxes[np.random.choice(num_boxes, k, replace=False)]
    
    while True:
        # Calculate IoU distances
        for i in range(num_boxes):
            distances[i] = 1 - iou_distance(boxes[i], clusters)
        
        # Assign boxes to nearest centroid
        nearest_clusters = np.argmin(distances, axis=1)
        
        # Check for convergence
        if (last_clusters == nearest_clusters).all():
            break
        
        # Update centroids
        for i in range(k):
            clusters[i] = np.mean(boxes[nearest_clusters == i], axis=0)
        
        last_clusters = nearest_clusters
    
    return clusters, nearest_clusters

def iou_distance(box, clusters):
    """
    Calculate IoU-based distance between a box and clusters
    """
    x = np.minimum(box[0], clusters[:, 0])
    y = np.minimum(box[1], clusters[:, 1])
    
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    
    iou = intersection / (box_area + cluster_area - intersection)
    
    return iou

def assign_anchors_to_scales(anchors, input_shape=(256, 128)):
    """
    Assign anchors to appropriate scales based on size
    """
    # Sort anchors by area
    flattened = anchors.reshape(-1, 2)
    areas = flattened[:, 0] * flattened[:, 1]
    sorted_indices = np.argsort(areas)
    sorted_anchors = flattened[sorted_indices]
    
    # Split into three groups for different detection scales
    num_anchors = len(sorted_anchors)
    anchors_per_scale = num_anchors // 3
    
    small_anchors = sorted_anchors[:anchors_per_scale]
    medium_anchors = sorted_anchors[anchors_per_scale:2*anchors_per_scale]
    large_anchors = sorted_anchors[2*anchors_per_scale:]
    
    # Reshape for multi-scale detection (smallest objects detected at highest resolution)
    return np.array([
        small_anchors.reshape(-1, 2),   # For small objects (high resolution)
        medium_anchors.reshape(-1, 2),  # For medium objects
        large_anchors.reshape(-1, 2)    # For large objects (low resolution)
    ])

def aspect_ratio_adjustment(anchors, dataset_stats=None):
    """
    Adjust anchors to better match aspect ratios in the dataset
    """
    if dataset_stats is None:
        # COCO dataset contains a diverse set of objects with these approximate aspect ratios
        # Width/height ratios for common COCO object categories
        person_ratio = 0.4      # People are taller than wide
        animal_ratio = 1.3      # Dogs, cats, etc. are typically wider than tall
        vehicle_ratio = 1.6     # Cars, buses (less extreme than BDD100K-specific vehicles)
        furniture_ratio = 1.5   # Tables, chairs, etc.
        electronics_ratio = 0.8  # TVs, laptops, etc.
        kitchenware_ratio = 1.0  # Cups, bottles (mostly square/round)
        sports_ratio = 3.0      # Baseball bats, tennis rackets (very wide aspect ratio)
        food_ratio = 1.1        # Pizza, sandwich, etc.
        
        # Small anchors (often small objects like sports equipment, kitchenware)
        anchors[0, 0] = adjust_anchor(anchors[0, 0], target_ratio=kitchenware_ratio)
        anchors[0, 1] = adjust_anchor(anchors[0, 1], target_ratio=electronics_ratio)
        anchors[0, 2] = adjust_anchor(anchors[0, 2], target_ratio=sports_ratio * 0.5)
        
        # Medium anchors (often people, animals, small furniture)
        anchors[1, 0] = adjust_anchor(anchors[1, 0], target_ratio=person_ratio)
        anchors[1, 1] = adjust_anchor(anchors[1, 1], target_ratio=animal_ratio)
        anchors[1, 2] = adjust_anchor(anchors[1, 2], target_ratio=furniture_ratio * 0.8)
        
        # Large anchors (often vehicles, large furniture)
        anchors[2, 0] = adjust_anchor(anchors[2, 0], target_ratio=furniture_ratio)
        anchors[2, 1] = adjust_anchor(anchors[2, 1], target_ratio=vehicle_ratio)
        anchors[2, 2] = adjust_anchor(anchors[2, 2], target_ratio=animal_ratio * 1.5)
    
    return anchors

def adjust_anchor(anchor, target_ratio):
    """
    Adjust anchor to match target aspect ratio while preserving area
    """
    w, h = anchor
    area = w * h
    
    # Calculate new dimensions with the same area but target aspect ratio
    new_w = np.sqrt(area * target_ratio)
    new_h = area / new_w
    
    return np.array([new_w, new_h])

def generate_resolution_adaptive_anchors(input_size=(256, 128), base_anchors=None):
    """
    Generate anchors that adapt to different input resolutions
    """
    # Base anchors optimized for 416×416 (standard YOLO resolution)
    if base_anchors is None:
        base_anchors = np.array([
            # Small anchors
            [[10, 13], [16, 30], [33, 23]],
            # Medium anchors
            [[30, 61], [62, 45], [59, 119]],
            # Large anchors
            [[116, 90], [156, 198], [373, 326]]
        ], dtype=np.float32)  # Add explicit float type here
    else:
        # Ensure base_anchors is float type
        base_anchors = base_anchors.astype(np.float32)
    
    # Standard reference size
    ref_width, ref_height = 416, 416
    target_width, target_height = input_size
    
    # Scale anchors based on resolution ratio
    width_ratio = target_width / ref_width
    height_ratio = target_height / ref_height
    
    scaled_anchors = base_anchors.copy()
    scaled_anchors[:, :, 0] *= width_ratio
    scaled_anchors[:, :, 1] *= height_ratio
    
    return scaled_anchors

def generate_anchors(dataset=None, input_size=(256, 128), method='kmeans'):
    """
    Generate optimized anchor boxes for object detection
    
    Args:
        dataset: Dataset to analyze for optimal anchor boxes
        input_size: Input resolution (width, height)
        method: Method to use ('kmeans', 'default', or 'adaptive')
        
    Returns:
        Array of anchor boxes grouped by scale
    """
    if method == 'kmeans' and dataset is not None:
        # Generate anchors using k-means clustering
        anchors = generate_optimized_anchors(9, dataset)
        # Assign to scales based on size
        anchors = assign_anchors_to_scales(anchors, input_size)
        # Adjust aspect ratios
        anchors = aspect_ratio_adjustment(anchors)
    elif method == 'adaptive':
        # Generate resolution-adaptive anchors
        anchors = generate_resolution_adaptive_anchors(input_size)
    else:
        # Default manually specified anchors
        anchors = np.array([
            # Small objects (traffic lights, distant cars)
            [[8, 16], [16, 12], [24, 20]],
            # Medium objects (nearby cars, trucks)
            [[32, 24], [48, 30], [64, 48]],
            # Large objects (nearby buses, trucks, close-up vehicles)
            [[96, 56], [128, 80], [192, 112]]
        ])
        
    return anchors
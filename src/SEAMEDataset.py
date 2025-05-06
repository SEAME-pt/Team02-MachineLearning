import os
import json
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from src.augmentation import LaneDetectionAugmentation
from torch.utils.data import Dataset


def get_binary_labels(height, width, pts, thickness=5):
    """ Get the binary labels. this function is similar to
    @get_binary_image, but it returns labels in 2 x H x W format
    this label will be used in the CrossEntropyLoss function.

    Args:
        img: numpy array
        pts: set of lanes, each lane is a set of points

    Output:

    """
    bin_img = np.zeros(shape=[height, width], dtype=np.uint8)
    for lane in pts:
        cv2.polylines(
            bin_img,
            np.int32(
                [lane]),
            isClosed=False,
            color=255,
            thickness=thickness)

    bin_labels = np.zeros_like(bin_img, dtype=bool)
    bin_labels[bin_img != 0] = True
    bin_labels = np.stack([~bin_labels, bin_labels]).astype(np.uint8)
    return bin_labels

def get_instance_labels(height, width, pts, thickness=5, max_lanes=5):
    """  Get the instance segmentation labels.
    this function is similar to @get_instance_image,
    but it returns label in L x H x W format

    Args:
            image
            pts

    Output:
            max Lanes x H x W, number of actual lanes
    """
    if len(pts) > max_lanes:
        pts = pts[:max_lanes]

    ins_labels = np.zeros(shape=[0, height, width], dtype=np.uint8)

    n_lanes = 0
    for lane in pts:
        ins_img = np.zeros(shape=[height, width], dtype=np.uint8)
        cv2.polylines(
            ins_img,
            np.int32(
                [lane]),
            isClosed=False,
            color=1,
            thickness=thickness)

        # there are some cases where the line could not be draw, such as one
        # point, we need to remove these cases
        # also, if there is overlapping among lanes, only the latest lane is
        # labeled
        if ins_img.sum() != 0:
            # comment this line because it will zero out previous lane data,
            # this leads to NaN error in computing the discriminative loss

            # ins_labels[:, ins_img != 0] = 0
            ins_labels = np.concatenate([ins_labels, ins_img[np.newaxis]])
            n_lanes += 1

    if n_lanes < max_lanes:
        n_pad_lanes = max_lanes - n_lanes
        pad_labels = np.zeros(
            shape=[
                n_pad_lanes,
                height,
                width],
            dtype=np.uint8)
        ins_labels = np.concatenate([ins_labels, pad_labels])

    return ins_labels, n_lanes

def get_image_transform():
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    t = [transforms.ToTensor(),
         normalizer]

    transform = transforms.Compose(t)
    return transform

class SEAMEDataset(Dataset):
    def __init__(self, json_paths, img_dir, width=512, height=256, is_train=True, thickness=5):
        """
        TuSimple Dataset for lane detection
        
        Args:
            json_paths: List of json files containing lane annotations
            img_dir: Directory containing the images
            width: Target image width
            height: Target image height
            is_train: Whether this is for training (enables augmentations)
            thickness: Thickness of lane lines in the binary mask
        """
        self.width = width
        self.height = height
        self.thickness = thickness
        self.img_dir = img_dir
        self.transform = get_image_transform()
        self.is_train = is_train

        # Initialize augmentation
        self.augmentation = LaneDetectionAugmentation(
            height=height, 
            width=width,
        )
        
        # Load all samples from all json files
        self.samples = []
        for json_path in json_paths:
            with open(json_path, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples with augmentation {'enabled' if is_train else 'disabled'}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        file_path = os.path.join(self.img_dir, info['raw_file'])
        
        # Read and resize image
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not read image: {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        width_org = image.shape[1]
        height_org = image.shape[0]
        image = cv2.resize(image, (self.width, self.height))

        # Process lane points
        x_lanes = info['lanes']
        y_samples = info['h_samples']
        
        # Create points list with list comprehension
        pts = [
            [(x, y) for (x, y) in zip(lane, y_samples) if x >= 0]
            for lane in x_lanes
        ]

        # Remove empty lanes
        pts = [l for l in pts if len(l) > 0]

        # Calculate scaling rates
        x_rate = 1.0 * self.width / width_org
        y_rate = 1.0 * self.height / height_org

        # Scale points
        pts = [[(int(round(x*x_rate)), int(round(y*y_rate)))
                for (x, y) in lane] for lane in pts]

        bin_labels = get_binary_labels(self.height, self.width, pts, thickness=self.thickness)
        instance_labels, n_lanes = get_instance_labels(self.height, self.width, pts, thickness=self.thickness, max_lanes=4)

        if self.is_train:
            image, bin_labels, instance_labels = self.augmentation(image, bin_labels, instance_labels)
            return image, bin_labels, instance_labels, n_lanes
        else:
            image = self.transform(image)
            bin_labels = torch.Tensor(bin_labels)
            instance_labels = torch.Tensor(instance_labels)
            return image, bin_labels, instance_labels, n_lanes
import torch
import torch.nn as nn
import torchvision.models.segmentation as models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

class LaneSegmentationModel(nn.Module):
    def __init__(self, num_classes=1):
        super(LaneSegmentationModel, self).__init__()
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = models.deeplabv3_resnet50(weights=weights)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

class SpatialCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(SpatialCNN, self).__init__()
        
        # MobileNetV2 backbone
        mobilenet = mobilenet_v2(weights="DEFAULT")  # Updated from pretrained=True
        self.backbone = nn.Sequential(*list(mobilenet.features.children())[:14])
        
        # SCNN specific layers - message passing between adjacent rows and columns
        self.message_passing = SimplifiedMessagePass(96)
        
        # Decoder layers to upsample features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Add one more upsampling layer to match the target size
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply message passing
        processed_features = self.message_passing(features)
        
        # Decode features to original resolution
        output = self.decoder(processed_features)
        
        return output


class MessagePassingBlock(nn.Module):
    """
    SCNN message passing module that performs spatial message passing
    in four directions: top-to-bottom, bottom-to-top, left-to-right, right-to-left
    """
    def __init__(self, channels):
        super(MessagePassingBlock, self).__init__()
        self.channels = channels
        
        # Define convolutions for each direction
        self.down_conv = nn.Conv2d(channels, channels, kernel_size=(1, 9), padding=(0, 4))
        self.up_conv = nn.Conv2d(channels, channels, kernel_size=(1, 9), padding=(0, 4))
        self.right_conv = nn.Conv2d(channels, channels, kernel_size=(9, 1), padding=(4, 0))
        self.left_conv = nn.Conv2d(channels, channels, kernel_size=(9, 1), padding=(4, 0))
    
    def _down_to_up(self, x):
        height = x.size()[2]
        top = x[:, :, 0:1, :]
        # Create a new tensor instead of modifying in-place
        result = top.clone()
        
        for i in range(1, height):
            # Process the previous result
            prev = result
            current = x[:, :, i:i+1, :]
            conv_prev = F.relu(self.up_conv(prev))
            # Concatenate along height dimension
            result = torch.cat([result, current + conv_prev], dim=2)
            
        return result
    
    def _up_to_down(self, x):
        height = x.size()[2]
        bottom = x[:, :, -1:, :]
        # Create a new tensor instead of modifying in-place
        result = bottom.clone()
        
        for i in range(height-2, -1, -1):
            # Start from the bottom and work upwards
            prev = result
            current = x[:, :, i:i+1, :]
            conv_prev = F.relu(self.down_conv(prev))
            # Add at the beginning of tensor
            result = torch.cat([current + conv_prev, result], dim=2)
            
        return result
    
    def _right_to_left(self, x):
        width = x.size()[3]
        leftmost = x[:, :, :, 0:1]
        # Create a new tensor instead of modifying in-place
        result = leftmost.clone()
        
        for i in range(1, width):
            # Process from left to right
            prev = result
            current = x[:, :, :, i:i+1]
            conv_prev = F.relu(self.left_conv(prev))
            # Concatenate along width dimension
            result = torch.cat([result, current + conv_prev], dim=3)
            
        return result
    
    def _left_to_right(self, x):
        width = x.size()[3]
        rightmost = x[:, :, :, -1:]
        # Create a new tensor instead of modifying in-place
        result = rightmost.clone()
        
        for i in range(width-2, -1, -1):
            # Process from right to left
            prev = result
            current = x[:, :, :, i:i+1]
            conv_prev = F.relu(self.right_conv(prev))
            # Add at the beginning of tensor
            result = torch.cat([current + conv_prev, result], dim=3)
            
        return result
    
    def forward(self, x):
        # Perform message passing in four directions
        down_up = self._down_to_up(x)
        up_down = self._up_to_down(x)
        right_left = self._right_to_left(x)
        left_right = self._left_to_right(x)
        
        # Combine results from all directions through addition
        # Ensure all tensors have the same shape before adding
        return x + down_up + up_down + right_left + left_right

class SimplifiedMessagePass(nn.Module):
    """
    A simplified message passing module that uses standard convolutions
    for better compatibility with MPS device
    """
    def __init__(self, channels):
        super(SimplifiedMessagePass, self).__init__()
        
        # Vertical passes (down and up)
        self.vertical_conv1 = nn.Conv2d(channels, channels, kernel_size=(5, 1), 
                                       padding=(2, 0), bias=False)
        self.vertical_conv2 = nn.Conv2d(channels, channels, kernel_size=(5, 1), 
                                       padding=(2, 0), bias=False)
        
        # Horizontal passes (left and right)
        self.horizontal_conv1 = nn.Conv2d(channels, channels, kernel_size=(1, 5), 
                                         padding=(0, 2), bias=False)
        self.horizontal_conv2 = nn.Conv2d(channels, channels, kernel_size=(1, 5), 
                                         padding=(0, 2), bias=False)
        
        # Add normalization layers
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.norm3 = nn.BatchNorm2d(channels)
        self.norm4 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        # First vertical pass
        out = x
        out = F.relu(self.norm1(self.vertical_conv1(out)))
        out = F.relu(self.norm1(self.vertical_conv1(out)))
        out = F.relu(self.norm1(self.vertical_conv1(out)))
        
        # First horizontal pass
        out = F.relu(self.norm2(self.horizontal_conv1(out)))
        out = F.relu(self.norm2(self.horizontal_conv1(out)))
        out = F.relu(self.norm2(self.horizontal_conv1(out)))
        
        # Second vertical pass
        out = F.relu(self.norm3(self.vertical_conv2(out)))
        out = F.relu(self.norm3(self.vertical_conv2(out)))
        out = F.relu(self.norm3(self.vertical_conv2(out)))
        
        # Second horizontal pass
        out = F.relu(self.norm4(self.horizontal_conv2(out)))
        out = F.relu(self.norm4(self.horizontal_conv2(out)))
        out = F.relu(self.norm4(self.horizontal_conv2(out)))
        
        # Skip connection
        return x + out


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

class MobileNetV2UNet(nn.Module):
    def __init__(self, output_channels=1):
        super(MobileNetV2UNet, self).__init__()
        
        # Load pre-trained MobileNetV2 backbone
        self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        
        # Get feature layers from MobileNetV2
        self.down1 = self.backbone.features[:2]    # Output: 16 channels
        self.down2 = self.backbone.features[2:4]   # Output: 24 channels
        self.down3 = self.backbone.features[4:7]   # Output: 32 channels
        self.down4 = self.backbone.features[7:14]  # Output: 96 channels
        self.down5 = self.backbone.features[14:]   # Output: 320 channels
        
        # Upsampling path
        self.up1 = up(320 + 96, 96)
        self.up2 = up(96 + 32, 32)
        self.up3 = up(32 + 24, 24)
        self.up4 = up(24 + 16, 16)
        
        # Output layer
        self.outc = outconv(16, output_channels)
        
    def forward(self, x):
        # Downsampling through backbone
        x1 = self.down1(x)  # 1/2
        x2 = self.down2(x1)  # 1/4
        x3 = self.down3(x2)  # 1/8
        x4 = self.down4(x3)  # 1/16
        x5 = self.down5(x4)  # 1/32
        
        # Upsampling and concatenation with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final output
        x = self.outc(x)
        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        # self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//2, 1),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch//2, out_ch, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, base_filters=64):
        super(UNet, self).__init__()
        self.inc = inconv(3, base_filters)
        self.down1 = down(base_filters, base_filters*2)
        self.down2 = down(base_filters*2, base_filters*4)
        self.down3 = down(base_filters*4, base_filters*4)

        self.up1 = up(base_filters*8, base_filters*2)
        self.up2 = up(base_filters*4, base_filters)
        self.up3 = up(base_filters*2, base_filters)
        self.sem_out = outconv(base_filters, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        sem = self.sem_out(x)
        return sem

class LightUNet(nn.Module):
    def __init__(self, base_filters=32):
        super(LightUNet, self).__init__()
        self.inc = inconv(3, base_filters)
        self.down1 = down(base_filters, base_filters*2)
        self.down2 = down(base_filters*2, base_filters*4)
        self.down3 = down(base_filters*4, base_filters*4)
        
        self.up1 = up(base_filters*8, base_filters*2)
        self.up2 = up(base_filters*4, base_filters)
        self.up3 = up(base_filters*2, base_filters)
        self.sem_out = outconv(base_filters, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        sem = self.sem_out(x)
        return sem
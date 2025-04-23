import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import MobileNet_V2_Weights

class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and leaky ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class DetectionBlock(nn.Module):
    """Detection block that outputs predictions for a specific scale"""
    def __init__(self, in_channels, out_channels, num_anchors, num_classes):
        super(DetectionBlock, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        # Detection output: (x, y, w, h, obj_conf, classes)
        self.out_channels = num_anchors * (5 + num_classes)
        
        self.conv1 = ConvBlock(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, self.out_channels, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Get actual grid dimensions
        batch_size = x.size(0)
        grid_h = x.size(2)
        grid_w = x.size(3)
        
        # Reshape using actual grid dimensions
        prediction = x.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_h, grid_w)
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
        
        return prediction

class SimpleYOLO(nn.Module):
    """YOLO-like model with multi-scale detection using anchor boxes"""
    def __init__(self, num_classes, input_size=(256, 128), use_default_anchors=False):
        super(SimpleYOLO, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Generate anchors directly in the model initialization
        if use_default_anchors:
            anchors = self._get_default_anchors()
        else:
            anchors = self._generate_adaptive_anchors(input_size)
            
        self.register_buffer('anchors', anchors)
        self.num_anchors = self.anchors.size(1)
            
        # MobileNetV2 backbone setup
        backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        features = backbone.features
        self.backbone_layer1 = features[:7]   # First part (up to stride 8)
        self.backbone_layer2 = features[7:14] # Middle part (up to stride 16)
        self.backbone_layer3 = features[14:]  # Last part (up to stride 32)
        
        # Feature dimensions for MobileNetV2
        self.feature_dims = [32, 96, 1280]  # Adjust based on actual output dimensions

        # Detection layers (adjust channel numbers based on backbone)
        self.detect1 = DetectionBlock(self.feature_dims[2], 256, self.num_anchors, num_classes)
        
        # Upsampling and feature fusion
        self.up1 = nn.Sequential(
            ConvBlock(self.feature_dims[2], 128, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        
        # Adjust input channels based on concatenation
        self.detect2 = DetectionBlock(self.feature_dims[1] + 128, 128, self.num_anchors, num_classes)
        
        self.up2 = nn.Sequential(
            ConvBlock(self.feature_dims[1] + 128, 64, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        
        self.detect3 = DetectionBlock(self.feature_dims[0] + 64, 64, self.num_anchors, num_classes)
    
    def forward(self, x):
        # Backbone feature extraction
        f1 = self.backbone_layer1(x)
        f2 = self.backbone_layer2(f1)
        f3 = self.backbone_layer3(f2)
        
        # Detection at largest scale
        detect3 = self.detect1(f3)
        
        # Upsample and concatenate with medium features
        up = self.up1(f3)
        f2_cat = torch.cat([up, f2], dim=1)
        detect2 = self.detect2(f2_cat)
        
        # Upsample and concatenate with small features
        up = self.up2(f2_cat)
        f1_cat = torch.cat([up, f1], dim=1)
        detect1 = self.detect3(f1_cat)
        
        return [detect1, detect2, detect3]
    
    def _get_default_anchors(self):
        """Return default anchors for common object detection"""
        default_anchors = [
            # Small objects (traffic lights, distant cars)
            [[8, 16], [16, 12], [24, 20]],
            # Medium objects (nearby cars, trucks)
            [[32, 24], [48, 30], [64, 48]],
            # Large objects (nearby buses, trucks, close-up vehicles)
            [[96, 56], [128, 80], [192, 112]]
        ]
        return torch.tensor(default_anchors).float().view(-1, 3, 2)
        
    def _generate_adaptive_anchors(self, input_size):
        """Generate resolution-adaptive anchors based on input size"""
        # Base anchors for 416×416
        base_anchors = [
            # Small anchors
            [[10, 13], [16, 30], [33, 23]],
            # Medium anchors
            [[30, 61], [62, 45], [59, 119]],
            # Large anchors
            [[116, 90], [156, 198], [373, 326]]
        ]
        
        # Scale anchors based on resolution ratio
        width, height = input_size
        width_ratio = width / 416
        height_ratio = height / 416
        
        scaled_anchors = []
        for scale_anchors in base_anchors:
            scaled_scale = []
            for w, h in scale_anchors:
                scaled_w = w * width_ratio
                scaled_h = h * height_ratio
                scaled_scale.append([scaled_w, scaled_h])
            scaled_anchors.append(scaled_scale)
            
        return torch.tensor(scaled_anchors).float()
    
    def predict_boxes(self, detections, input_dim, conf_thresh=0.5):
        """
        Convert raw detector outputs to bounding boxes
        """
        batch_size = detections[0].size(0)
        all_boxes = []
        
        for i in range(batch_size):
            boxes = []
            
            # Process each scale
            for scale, detection in enumerate(detections):
                # Get anchors for this scale
                anchors = self.anchors[scale]
                
                # Get grid dimensions
                grid_h, grid_w = detection.size(2), detection.size(3)
                
                # Extract prediction for this batch item
                pred = detection[i]  # Shape [3, 16, 32, 11]
                
                # Create grid with exact same dimensions as pred[...,0]
                grid_y = torch.arange(grid_h, device=detection.device).view(1, grid_h, 1).repeat(self.num_anchors, 1, grid_w)
                grid_x = torch.arange(grid_w, device=detection.device).view(1, 1, grid_w).repeat(self.num_anchors, grid_h, 1)
                
                # Apply sigmoid to center coordinates and confidence
                pred[..., 0:2] = torch.sigmoid(pred[..., 0:2])
                pred[..., 4:] = torch.sigmoid(pred[..., 4:])
                
                # Calculate bounding box coordinates - grid_x and pred now have compatible shapes
                pred[..., 0] = pred[..., 0] + grid_x  # x center
                pred[..., 1] = pred[..., 1] + grid_y  # y center
                
                # Apply anchors to width and height
                anchors = anchors.to(detection.device)
                anchor_w = anchors[:, 0].view(self.num_anchors, 1, 1)
                anchor_h = anchors[:, 1].view(self.num_anchors, 1, 1)
                
                pred[..., 2] = torch.exp(pred[..., 2]) * anchor_w  # width
                pred[..., 3] = torch.exp(pred[..., 3]) * anchor_h  # height
                
                
                # Convert to x1, y1, x2, y2 format
                pred_boxes = pred[..., :4].clone()
                pred_boxes[..., 0] = pred[..., 0] - pred[..., 2] / 2  # x1
                pred_boxes[..., 1] = pred[..., 1] - pred[..., 3] / 2  # y1
                pred_boxes[..., 2] = pred[..., 0] + pred[..., 2] / 2  # x2
                pred_boxes[..., 3] = pred[..., 1] + pred[..., 3] / 2  # y2
                
                # Scale to original image dimensions
                # Use separate scaling factors for width and height
                stride_h = input_dim / grid_h
                stride_w = input_dim * 2 / grid_w  # Assuming width is 2× height
                
                pred_boxes[..., 0] *= stride_w  # Scale x coordinates
                pred_boxes[..., 2] *= stride_w
                pred_boxes[..., 1] *= stride_h  # Scale y coordinates
                pred_boxes[..., 3] *= stride_h
                
                # Get confidence and class predictions
                obj_conf = pred[..., 4:5]
                cls_conf, cls_idx = torch.max(pred[..., 5:], dim=-1, keepdim=True)
                
                # Filter by confidence
                conf_mask = (obj_conf * cls_conf).squeeze(-1) >= conf_thresh
                
                # Extract valid predictions
                for a in range(self.num_anchors):
                    for h in range(grid_h):
                        for w in range(grid_w):
                            if conf_mask[a, h, w]:
                                box = torch.cat([
                                    pred_boxes[a, h, w],
                                    obj_conf[a, h, w],
                                    cls_conf[a, h, w],
                                    cls_idx[a, h, w].float()
                                ])
                                boxes.append(box)
            
            # If no boxes, add empty tensor
            if len(boxes) == 0:
                all_boxes.append(torch.zeros((0, 7)).to(detections[0].device))
            else:
                all_boxes.append(torch.stack(boxes))
        
        return all_boxes
    
    def non_max_suppression(self, boxes, nms_thresh=0.45):
        """
        Apply non-maximum suppression to remove overlapping bounding boxes
        
        Args:
            boxes: Tensor of shape [N, 7] where each row is
                  [x1, y1, x2, y2, obj_conf, cls_conf, cls_idx]
            nms_thresh: IoU threshold for NMS
            
        Returns:
            Filtered boxes after NMS
        """
        if boxes.size(0) == 0:
            return boxes
        
        # Get box coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Calculate area
        area = (x2 - x1) * (y2 - y1)
        
        # Get final score (obj_conf * cls_conf)
        score = boxes[:, 4] * boxes[:, 5]
        
        # Sort by decreasing score
        _, order = score.sort(0, descending=True)
        
        keep = []
        while order.numel() > 0:
            # Pick the box with highest score
            if order.numel() == 1:
                keep.append(order.item())
                break
            
            i = order[0].item()
            keep.append(i)
            
            # Calculate IoU with remaining boxes
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            
            w = torch.max(torch.zeros_like(xx2), xx2 - xx1)
            h = torch.max(torch.zeros_like(yy2), yy2 - yy1)
            
            inter = w * h
            iou = inter / (area[i] + area[order[1:]] - inter)
            
            # Keep boxes with IoU < nms_thresh
            inds = torch.where(iou <= nms_thresh)[0]
            order = order[inds + 1]
        
        return boxes[keep]


class YOLOLoss(nn.Module):
    """YOLO loss function for object detection"""
    def __init__(self, anchors, num_classes, input_dim, device):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = anchors.size(1)
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.device = device
        
        # MSE loss for bounding box coordinates and size
        self.mse_loss = nn.MSELoss(reduction='sum')
        # BCE loss for objectness and class prediction
        self.bce_loss = nn.BCELoss(reduction='sum')
        # Classification loss weight
        self.cls_weight = 1.0
        # No object confidence loss weight (reduce penalty for background)
        self.no_obj_weight = 0.5
    
    def forward(self, predictions, targets):
        """
        Calculate YOLO loss
        
        Args:
            predictions: List of predictions from model
            targets: List of target tensors
        """
        total_loss = 0
        for scale, (prediction, anchors) in enumerate(zip(predictions, self.anchors)):
            # Get current grid dimensions
            batch_size = prediction.size(0)
            grid_h = prediction.size(2)  # Height of the grid
            grid_w = prediction.size(3)  # Width of the grid
            
            # Stride from original image to current grid
            stride_h = self.input_dim / grid_h
            stride_w = self.input_dim * 2 / grid_w   # Assuming input is square
            
            # Scale anchors to current grid size
            scaled_anchors_w = anchors[:, 0] * stride_w
            scaled_anchors_h = anchors[:, 1] * stride_h
            scaled_anchors = torch.stack((scaled_anchors_w, scaled_anchors_h), dim=1).to(self.device)
            
            # Create target tensors - now using grid_h and grid_w separately
            obj_mask = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, 
                                dtype=torch.bool, device=self.device)
            no_obj_mask = torch.ones(batch_size, self.num_anchors, grid_h, grid_w, 
                                    dtype=torch.bool, device=self.device)
            tx = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, 
                            device=self.device)
            ty = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, 
                            device=self.device)
            tw = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, 
                            device=self.device)
            th = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, 
                            device=self.device)
            tcls = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, self.num_classes, 
                            device=self.device)
            
            # Process target boxes
            for target in targets:
                # Skip invalid targets
                if target.size(0) == 0:
                    continue
                
                # Process each object in this target batch separately
                for obj_idx in range(target.size(0)):
                    # Extract target information for this specific object
                    b = target[obj_idx, 0].long().item()  # batch index
                    cls = target[obj_idx, 1].long().item()  # class index
                    
                    # Convert normalized coords to grid coordinates
                    # Use grid_w for x and grid_h for y to handle non-square grids
                    x = target[obj_idx, 2] * grid_w  # center x (grid units)
                    y = target[obj_idx, 3] * grid_h  # center y (grid units)
                    w = target[obj_idx, 4] * self.input_dim * 2  # width (pixel units, adjusted for aspect ratio)
                    h = target[obj_idx, 5] * self.input_dim   # height (pixel units)
                    
                    # Skip invalid targets (those outside the grid)
                    if x < 0 or y < 0 or x > grid_w-1 or y > grid_h-1:
                        continue
                        
                    # Convert to grid cell coordinates
                    gx = int(x)
                    gy = int(y)
                    
                    # Get fractional part of center coordinates
                    tx_target = x - gx
                    ty_target = y - gy
                    
                    # Find best anchor based on IoU
                    best_iou = 0
                    best_anchor = 0
                    
                    for i, (a_w, a_h) in enumerate(scaled_anchors):
                        # Calculate IoU between target and anchor
                        iou = self.bbox_iou(
                            torch.tensor([0, 0, w, h], device=self.device), 
                            torch.tensor([0, 0, a_w, a_h], device=self.device)
                        )
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_anchor = i
                    
                    # Update masks and targets using the best anchor
                    i = best_anchor
                    obj_mask[b, i, gy, gx] = True
                    no_obj_mask[b, i, gy, gx] = False
                    
                    # Update target values
                    tx[b, i, gy, gx] = tx_target
                    ty[b, i, gy, gx] = ty_target
                    tw[b, i, gy, gx] = torch.log(w / scaled_anchors[i, 0] + 1e-16)
                    th[b, i, gy, gx] = torch.log(h / scaled_anchors[i, 1] + 1e-16)
                    tcls[b, i, gy, gx, cls] = 1
            
            # Reshape prediction
            pred_box = prediction[..., :4].clone()
            pred_obj = prediction[..., 4]
            pred_cls = prediction[..., 5:]
            
            # Apply sigmoid to center coords, objectness, and class predictions
            pred_box[..., 0:2] = torch.sigmoid(pred_box[..., 0:2])
            pred_obj = torch.sigmoid(pred_obj)
            pred_cls = torch.sigmoid(pred_cls)
            
            # Calculate losses only if there are objects
            if obj_mask.sum() > 0:
                # Box coordinate loss (x, y)
                loss_x = self.mse_loss(pred_box[..., 0][obj_mask], tx[obj_mask])
                loss_y = self.mse_loss(pred_box[..., 1][obj_mask], ty[obj_mask])
                
                # Box size loss (w, h)
                loss_w = self.mse_loss(pred_box[..., 2][obj_mask], tw[obj_mask])
                loss_h = self.mse_loss(pred_box[..., 3][obj_mask], th[obj_mask])
                
                # Objectness loss
                loss_obj = self.bce_loss(pred_obj[obj_mask], torch.ones_like(pred_obj[obj_mask]))
                
                # Class prediction loss
                loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            else:
                loss_x = torch.tensor(0.0, device=self.device)
                loss_y = torch.tensor(0.0, device=self.device)
                loss_w = torch.tensor(0.0, device=self.device)
                loss_h = torch.tensor(0.0, device=self.device)
                loss_obj = torch.tensor(0.0, device=self.device)
                loss_cls = torch.tensor(0.0, device=self.device)
            
            # No object loss - always calculated
            loss_no_obj = self.bce_loss(pred_obj[no_obj_mask], torch.zeros_like(pred_obj[no_obj_mask]))
            
            # Total loss for this scale
            scale_loss = loss_x + loss_y + loss_w + loss_h + loss_obj + self.no_obj_weight * loss_no_obj + self.cls_weight * loss_cls
            
            # Add to total loss
            total_loss += scale_loss
        
        return total_loss
    
    def bbox_iou(self, box1, box2):
        """
        Calculate IoU between two boxes [x1, y1, w, h]
        """
        # Convert to [x1, y1, x2, y2]
        b1_x1, b1_y1 = box1[0], box1[1]
        b1_x2, b1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        b2_x1, b2_y1 = box2[0], box2[1]
        b2_x2, b2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Intersection area
        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        # Union area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        # IoU
        return inter_area / (b1_area + b2_area - inter_area + 1e-16)


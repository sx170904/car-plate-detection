##############################################################
# Coded By: Ng Kai Jiun (model architecture, from notebook)
##############################################################
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision.ops import nms


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True, freeze_early=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.out_channels = 512
        if freeze_early:
            for p in self.stem.parameters():   p.requires_grad = False
            for p in self.layer1.parameters(): p.requires_grad = False
            for p in self.layer2.parameters(): p.requires_grad = False

    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return x


class DetectionHead(nn.Module):
    def __init__(self, in_channels=512, grid_size=7):
        super().__init__()
        self.grid_size = grid_size
        self.neck = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.predictor = nn.Conv2d(128, 5, kernel_size=1)
        cols = torch.arange(grid_size).view(1, 1, grid_size).expand(1, grid_size, grid_size)
        rows = torch.arange(grid_size).view(1, grid_size, 1).expand(1, grid_size, grid_size)
        grid_xy = torch.stack([cols, rows], dim=-1).float()
        self.register_buffer('grid_xy', grid_xy)

    def forward(self, feat):
        x = self.neck(feat); x = self.predictor(x); x = x.permute(0, 2, 3, 1)
        tx_ty = torch.sigmoid(x[..., 0:2])
        tw_th = torch.sigmoid(x[..., 2:4])
        conf  = torch.sigmoid(x[..., 4:5])
        cxcy  = (self.grid_xy + tx_ty) / self.grid_size
        return torch.cat([cxcy, tw_th, conf], dim=-1)


class SSPD(nn.Module):
    def __init__(self, pretrained=True, freeze_early=True, grid_size=7):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained, freeze_early)
        self.head = DetectionHead(self.backbone.out_channels, grid_size)
        self.grid_size = grid_size

    def forward(self, images):
        return self.head(self.backbone(images))

    @torch.no_grad()
    def decode(self, images, conf_threshold=0.5, iou_threshold=0.4, max_boxes=10):
        self.eval()
        pred = self(images)
        B, S, _, _ = pred.shape
        pred_flat = pred.view(B, S * S, 5)
        results = []
        for b in range(B):
            boxes = pred_flat[b]
            boxes = boxes[boxes[:, 4] >= conf_threshold]
            if boxes.numel() == 0:
                results.append(torch.empty((0, 5), device=boxes.device)); continue
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            xyxy = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)
            keep = nms(xyxy, boxes[:, 4], iou_threshold)[:max_boxes]
            results.append(boxes[keep])
        return results

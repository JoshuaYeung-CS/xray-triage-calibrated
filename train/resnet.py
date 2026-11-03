"""ResNet-18 written out in plain PyTorch, with the same parameter names as torchvision's, so the
published ImageNet checkpoint (resnet18-f37072fd.pth) loads without installing torchvision."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + idt)


class ResNet18Trunk(nn.Module):
    """Everything up to (not including) global pooling; returns the 512-channel feature map."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))

    def load_imagenet(self, path: str) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        self.load_state_dict(sd, strict=True)


class TriageNet(nn.Module):
    """Input: (N, 1, 128, 128) grayscale in [0, 1]. Output: (logit (N,), cam (N, 7, 7)).

    Upsampling to 224 px and ImageNet normalisation happen inside the graph, so the exported ONNX
    file needs nothing but the raw image.
    """

    def __init__(self, input_size: int = 224):
        super().__init__()
        self.input_size = input_size
        self.trunk = ResNet18Trunk()
        self.head = nn.Linear(512, 1)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def features(self, x):
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        return self.trunk(x)

    def forward(self, x):
        a = self.features(x)  # (N, 512, h, w)
        w = self.head.weight[0]  # (512,)
        cam = torch.einsum("k,nkhw->nhw", w, a)
        logit = cam.mean(dim=(1, 2)) + self.head.bias[0]
        return logit, cam

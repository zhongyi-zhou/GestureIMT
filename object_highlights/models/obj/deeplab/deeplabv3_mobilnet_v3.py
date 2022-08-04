from typing import List, Optional
import torch
from torch import nn
from torch.nn import functional as F
# import torchvision
# from torchvision.models.segmentation import DeepLabV3
# from torchvision.models.segmentation.fcn import FCNHead
# from torchvision.models.feature_extraction import create_feature_extractor
from .mobilenetv3 import mobilenet_v3_small_multi_input
from collections import OrderedDict

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

def deeplab_mobilenetv3_multi_input(num_classes):
    backbone = mobilenet_v3_small_multi_input()
    index = backbone.features_2.__len__() - 1   
    out_inplanes = backbone.features_2[index].out_channels
    classifier = DeepLabHead(out_inplanes, num_classes)
    return DeeplabMobilenetv3(backbone, classifier)

class DeeplabMobilenetv3(nn.Module):
    def __init__(self, backbone, classifer):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifer
    
    def forward(self, x, addition_input):
        input_shape = x.shape[-2:]
        x = self.backbone.features_1(x)
        if addition_input is not None:
            x = x + self.backbone.addition_features(addition_input)
        x = self.backbone.features_2(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
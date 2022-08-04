import torch
from torch import nn
from models import human
import cv2
import numpy as np
import torchvision.transforms as transforms


class MyModel(nn.Module):
    def __init__(self, num_classes=20) -> None:
        super().__init__()
        self.human = human.resnet101(num_classes=num_classes, pretrained=None)
    
    def transform(self, x):
        # bgr
        transform = transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        return transform(x)

    def human_parse_postprocess(self, res, input_size):
        res = res[0][-1]
        upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
        res = upsample(res)
        res = torch.argmax(res, dim=1)
        bg = 1*(res == 0)
        hands = (res ==14) + (res==15)
        return bg, hands

    def forward(self, x):
        x = self.transform(x)
        input_size = x.shape[2:]
        x = self.human(x)
        bg, hands = self.human_parse_postprocess(x, input_size=input_size)
        return bg, hands
        # return x

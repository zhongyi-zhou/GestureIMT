import torch
from torch import nn
from models import human
import cv2
import numpy as np
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from collections import OrderedDict
import time
from torchvision.io import read_image
import torch.nn.functional as F

def resize_pad_tensor(img, size=[512,512]):
    h,w = img.shape[2:]
    if w/h>size[1]/size[0]:
        dw = int(size[1])
        dh = int(dw/w*h)
    else:
        dh = int(size[0])
        dw = int(dh/h*w)
    img_resize = torch.nn.functional.interpolate(img, size=(dh, dw), mode="bilinear", align_corners=True)
    w_pad = size[1] - dw
    h_pad = size[0] - dh
    return F.pad(img_resize, (w_pad//2, w_pad//2 + w_pad%2, h_pad//2, h_pad//2 + h_pad%2))


def resize_unpad_tensor(img, size=[480, 640]):
    if img.shape.__len__() == 3:
        img = img[None]
        h,w = img.shape[2:]
    elif img.shape.__len__() == 4:
        h,w = img.shape[2:]
    else:
        raise ValueError(f"expect img dim is either 3 or 4 but got {img.shape.__len__()}")
    if size[1]/size[0] > w/h:
        h_pad = int(h - w/size[1]*size[0])
        img = img[:,:,h_pad//2: -(h_pad//2 + h_pad%2), :]
    elif size[1]/size[0] < w/h:
        w_pad = int(w - h/size[0]*size[1])
        img = img[:,:, :, w_pad//2: -(w_pad//2 + w_pad%2)]
    return torch.nn.functional.interpolate(img, size=size, mode="bilinear", align_corners=True)

class ObjectInHand(nn.Module):
    def __init__(self, human_backbone="resnet101", obj_arch_backbone="unet_b0") -> None:
        super().__init__()
        if human_backbone == "resnet101":
            self.human = human.resnet101(num_classes=20, pretrained=None)
        elif human_backbone == "resnet18":
            self.human = human.resnet18(num_classes=20, pretrained=None)
        elif human_backbone == "efficientb0":
            self.human = human.efficientb0(num_classes=20, pretrained=None)
        else:
            raise ValueError(f"expect 'resnet101','resnet18' or 'efficientb0' of human_backbone but got {human_backbone}")
        if obj_arch_backbone.split("_")[0] == "unet":
            self.obj = smp.Unet(encoder_name=f"efficientnet-{obj_arch_backbone.split('_')[1]}", encoder_weights=None, in_channels=4, classes=1)   
        elif obj_arch_backbone.split("_")[0] == "unet++":
            self.obj = smp.UnetPlusPlus(encoder_name=f"efficientnet-{obj_arch_backbone.split('_')[1]}", encoder_weights=None, in_channels=4, classes=1)   
        elif obj_arch_backbone.split("_")[0] == "deeplab":
            self.obj = smp.DeepLabV3(encoder_name=f"efficientnet-{obj_arch_backbone.split('_')[1]}", encoder_weights=None, in_channels=4, classes=1)   
        elif obj_arch_backbone.split("_")[0] == "deeplab+":
            self.obj = smp.DeepLabV3Plus(encoder_name=f"efficientnet-{obj_arch_backbone.split('_')[1]}", encoder_weights=None, in_channels=4, classes=1)   

    def transform(self, x):
        # bgr
        trans = torch.nn.Sequential(
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),
        )
        return trans(x)

    def load_model(self, ckpt_human, ckpt_obj):
        def load_remove_module(model, my_state_dict):
            my_new_state_dict = OrderedDict()
            for k, v in my_state_dict.items():
                name = k[7:]  # remove `module.`
                my_new_state_dict[name] = v
            model.load_state_dict(my_new_state_dict)
            return model

        human_dict = torch.load(ckpt_human)["state_dict"] 
        obj_dict = torch.load(ckpt_obj)["state_dict"]
        self.human = load_remove_module(self.human, human_dict)
        self.obj = load_remove_module(self.obj, obj_dict)
        # self.obj.load_state_dict(obj_dict)

    
    def human_parse_postprocess(self, res, input_size):
        res = res[0][-1]
        res = resize_unpad_tensor(res)
        res = torch.argmax(res, dim=1)
        bg = 1*(res == 0)
        hands = (res ==14) + (res==15)
        return bg, hands

    def quick_postprocess(self, res, input_size):
        res = res[0][-1]
        res = torch.argmax(res, dim=1)
        hands = (res ==14) + (res==15)
        return resize_unpad_tensor(hands, input_size)


    def warm_up_forward(self):
        self.forward(torch.rand(1,3,480,640).cuda())

    def forward(self, x, trans=True, count_time=False):
        h,w = x.shape[2:]
        if count_time:
            t0 = time.time()
        if trans:
            x = self.transform(x)
        input_size = x.shape[2:]
        hands = self.human(resize_pad_tensor(x, (473, 473)))
        _, hands = self.human_parse_postprocess(hands, input_size=input_size)
        x = torch.cat((x, hands[None]), dim=1)
        x = self.obj(x) > 0
        return x, hands

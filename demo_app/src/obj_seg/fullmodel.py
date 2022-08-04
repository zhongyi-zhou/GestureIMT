import torch
from torch import nn
from src.obj_seg import human
import cv2
import numpy as np
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from collections import OrderedDict
import time
from collections import deque
# from torchvision.io import read_image
import torch.nn.functional as F
import random

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

class ObjectInHand(nn.Module):
    def __init__(self, arch="resnet18", num_classes=20, tcount=False) -> None:
        super().__init__()
        assert arch in ["resnet18", "resnet101"]
        if arch == "resnet101":
            self.human = human.resnet101(num_classes=num_classes, pretrained=None)
        elif arch == "resnet18":
            self.human = human.resnet18(num_classes=num_classes, pretrained=None)
        self.obj = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=4, classes=1)   
        
        if tcount:
            self.time_count = {
                "pre_process": deque([], maxlen=100),
                "hand": deque([], maxlen=100),
                "hand1": deque([], maxlen=100),
                "hand2": deque([], maxlen=100),
                "obj": deque([], maxlen=100),
                "all": deque([], maxlen=100)
            }
        else:
            self.time_count = None
    def transform(self, x, bgr=False):
        if bgr:
            # bgr
            trans = torch.nn.Sequential(
                transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),
                
            )
        else:
            # rgb
            trans = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),     
            ])
        return trans(x)

    def load_model(self, ckpt_human, ckpt_obj, remove_module=True):
        def load_remove_module(model, my_state_dict):
            my_new_state_dict = OrderedDict()
            for k, v in my_state_dict.items():
                name = k[7:]  # remove `module.`
                my_new_state_dict[name] = v
            model.load_state_dict(my_new_state_dict)
            return model

        human_dict = torch.load(ckpt_human)["state_dict"] 
        obj_dict = torch.load(ckpt_obj)["state_dict"]
        if remove_module:
            self.human = load_remove_module(self.human, human_dict)
            self.obj = load_remove_module(self.obj, obj_dict)

        else:
            self.human.load_state_dict(human_dict)
            self.obj.load_state_dict(obj_dict)
    
    def human_parse_postprocess(self, res, input_size):
        res = res[0][-1]
        upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
        res = upsample(res)
        res = torch.argmax(res, dim=1)
        bg = 1*(res == 0)
        hands = (res ==14) + (res==15)
        return bg, hands

    def quick_postprocess(self, res, input_size):
        res = res[0][-1]
        res = torch.argmax(res, dim=1)
        hands = (res ==14) + (res==15)
        # print(hands.shape)
        return F.interpolate(hands.float().unsqueeze(0), size=input_size)[0]

    def time_step(self, name, print_time=False):
        if name == "start":
            self.t0 = time.time()
            self.t_prev = time.time()
            return
        else:
            if name == "all":
                self.time_count[name].append(time.time()- self.t0)
            else:
                self.time_count[name].append(time.time()- self.t_prev)
            self.t_prev = time.time()
            if print_time:
                print(f"{name}: {round(sum(self.time_count[name])/len(self.time_count[name]), 4)}s")
        

    def warm_up_forward(self):
        self.forward(torch.rand(1,3,480,640).cuda())

    def forward(self, x):
        print_time=random.random() <0.02
        if self.time_count is not None:
            self.time_step("start")
        
        x = self.transform(x)
        input_size = x.shape[2:]
        if self.time_count is not None:
            self.time_step("pre_process", print_time)
        
        hands = self.human(resize_pad_tensor(x, (473, 473)))

        if self.time_count is not None:
            self.time_step("hand1", print_time)
        _, hands = self.human_parse_postprocess(hands, input_size=input_size)
        if self.time_count is not None:
            self.time_step("hand2", print_time)
        x = torch.cat((x, hands.unsqueeze(0)), dim=1)
        x = self.obj(x) > 0
        if x.shape[2:] != (480, 640):
            x = torch.nn.functional.interpolate(x.float(), (480, 640), mode="bilinear")
        if self.time_count is not None:
            self.time_step("obj", print_time)
            self.time_step("all", print_time)
        return x, hands
        # return x

    def forward_soft(self, x, slope=4):
        x = self.transform(x)
        input_size = x.shape[2:]
        hands = self.human(resize_pad_tensor(x, (473, 473)))
        _, hands = self.human_parse_postprocess(hands, input_size=input_size)
        x = torch.cat((x, hands.unsqueeze(0)), dim=1)
        x = self.obj(x) 
        x = (slope*x+1).clip(0, 1)
        return x, hands
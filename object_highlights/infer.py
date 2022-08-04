import os
from os.path import join
import torch
import cv2
from models.mymodel import MyModel
from collections import OrderedDict
import numpy as np
from models.obj.deeplab.deeplabv3_mobilnet_v3 import deeplab_mobilenetv3_multi_input
from models.obj.efficientunet import get_efficientunet_b0
import argparse
from base import Vis
from utils.utils import *

# ckptpath = "ckpt/exp-schp-201908301523-atr.pth"
# imgpath = "imgs/cup_example.jpg"

# model = MyModel()


# objnet = deeplab_mobilenetv3_multi_input(num_classes=1)
# objnet_ckpt = "ckpt/object/deeplabv3_mobilnetv3.pt"


# state_dict = torch.load(ckptpath)["state_dict"] 
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:]  # remove `module.`
#     new_state_dict[name] = v
# model.human.load_state_dict(new_state_dict)
# model.eval()

# img = cv2.imread(imgpath)[:,:,::-1] /255.
# img = cv2.imread(imgpath) /255.
# imgtensor = torch.from_numpy(img.copy()).permute(2,0,1).unsqueeze(0).float()

# bg, hands = model(imgtensor)


def load_model(model, ckptpath, remove_module=False):
    if remove_module:
        state_dict = torch.load(ckptpath)["state_dict"] 
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.human.load_state_dict(new_state_dict)
        model.eval()
        return model
    else:
        state_dict = torch.load(ckptpath)["model_state_dict"] 
        model.load_state_dict(state_dict)
        model.eval()
        return model

def saveimg(img, savepath):
    # img = np.array(tensor.squeeze()*1,dtype=np.uint8)
    cv2.imwrite(savepath, img)

# def resize_pad(img, size=[512,512]):
#     h,w,c = img.shape
#     if w>h:
#         dw = int(size[0])
#         dh = int(dw/w*h)
#     else:
#         dh = int(size[1])
#         dw = int(dh/h*w)
#     img_resize = cv2.resize(img, (dw, dh))
#     w_pad = size[0] - dw
#     h_pad = size[1] - dh
#     img_resize_pad = np.pad(img_resize, ((h_pad//2, h_pad//2 + h_pad%2), (w_pad//2, w_pad//2 + w_pad%2), (0,0)))
#     return img_resize_pad

# def resize_unpad(img, size):
#     img_resize = cv2.resize(img, (int(max(size)), int(max(size))))
#     if size[0]>size[1]:
#         h_unpad = size[0] - size[1]
#         return img_resize[h_unpad//2: h_unpad//2+size[1], :]
#     else:
#         w_unpad = size[1] - size[0]
#         return img_resize[:, w_unpad//2: w_unpad//2+size[0]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--handckpt', default="ckpt/exp-schp-201908261155-lip.pth", type=str,
                        help="hand segmentation model path")
    parser.add_argument('--objckpt', default="ckpt/object/unet_efficient-b0.pt", type=str,
                        help="object segmentation model path")
    parser.add_argument('--input', default="imgs/cup_example.jpg",
                        type=str, help='input image')
    parser.add_argument('--output', default="imgs/cup_example/",
                    type=str, help='output image dir')
    parser.add_argument('--vis_format', default="imgs/cup_example/",
                type=str, help='output image dir')
    args = parser.parse_args()
    
    handmodel = MyModel()
    handmodel = load_model(handmodel, args.handckpt, remove_module=True)

    objmodel = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=False)
    # objmodel = deeplab_mobilenetv3_multi_input(num_classes=1)
    objmodel = load_model(objmodel, args.objckpt, remove_module=False)

    img = cv2.imread(args.input) /255.

    vis = Vis(img*255)

    h,w,c = img.shape
    img_pad = resize_pad(img)
    imgtensor_bgr = torch.from_numpy(img_pad.copy()).permute(2,0,1).unsqueeze(0).float()
    imgtensor_rgb = torch.from_numpy(img[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0).float()

    bg, hands = handmodel(imgtensor_bgr)
    

    os.makedirs(args.output, exist_ok=True)
    handmask = resize_unpad(np.array(hands.permute(1,2,0)*1,dtype=np.uint8)*255, (w, h))
    cv2.imwrite(join(args.output, "handmask.png"), handmask)
    _ = vis.add_mask(handmask)
    
    resobj = objmodel(imgtensor_rgb, torch.from_numpy(handmask/255.).unsqueeze(0).unsqueeze(0).float())    
    objmask = np.array((resobj>0).squeeze(),dtype=np.uint8)*255
    saveimg(objmask, join(args.output, "objmask.png"))

    vis_final = vis.add_mask(objmask)
    saveimg(vis_final, join(args.output, "vis.png"))
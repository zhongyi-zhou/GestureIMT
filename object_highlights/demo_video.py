import os
from os.path import join
import torch
import cv2
from collections import OrderedDict
import numpy as np
import argparse
from base import Vis
from utils.utils import *
from tqdm import tqdm
from base.fullmodel import ObjectInHand

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
        state_dict = torch.load(ckptpath)["state_dict"] 
        model.load_state_dict(state_dict)
        model.eval()
        return model

def saveimg(img, savepath):
    cv2.imwrite(savepath, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--handckpt', default="ckpt/resnet18_adam.pth.tar", type=str,
                        help="hand segmentation model path")
    parser.add_argument('--handbackbone', default="resnet18", type=str)
    parser.add_argument('--objckpt', required=True, type=str,
                        help="object segmentation model path")
    parser.add_argument('--obj_model', default="unet_b0", type=str)
    parser.add_argument('--input', default="vids/tissue_video.mp4",
                        type=str, help='input video')
    parser.add_argument('--output', default="vids/tissue_out.mp4",
                    type=str, help='output image dir')
    args = parser.parse_args()
    
    model = ObjectInHand(human_backbone=args.handbackbone, obj_arch_backbone=args.obj_model)
    model.load_model(args.handckpt, args.objckpt)
    model.cuda()
    model.eval()

    cap = cv2.VideoCapture(args.input)
    _, frame = cap.read()
    H, W = frame.shape[:2]
    fps = cap.get(5)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vis = cv2.VideoWriter(args.output, fourcc, fps, (W,H))
    font = cv2.FONT_HERSHEY_SIMPLEX

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with torch.no_grad():
        for i in tqdm(range(length-1)):
            _, img = cap.read()
            h, w = img.shape[:2]
            vis = Vis(img)
            img = img/255.
            # the cuurent implement only support the ratio of 4:3, if you want to work with 16:9, for example,
            # you can do padding and resize by yourself. 
            img_resize = cv2.resize(img, (640, 480))
            img_norm = torch.from_numpy(img_resize.copy()).permute(2,0,1).unsqueeze(0).float().cuda()
            img_norm = model.transform(img_norm)
            hands = model.human(img_norm)
            _, hands = model.human_parse_postprocess(hands, input_size=[480, 640])
            mask = torch.cat((img_norm, hands.unsqueeze(0)), dim=1)
            mask = model.obj(mask) > 0
            mask = (mask[0].cpu().permute(1,2,0).numpy()*255).astype("uint8")[:,:,0]
            mask = cv2.resize(mask, (w, h))
            vis_final = vis.add_mask(mask)
            out_vis.write(vis_final)
    cap.release()
    out_vis.release()
    cv2.destroyAllWindows()
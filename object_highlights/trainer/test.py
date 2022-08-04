import os,sys
sys.path.append(os.getcwd())
from os.path import join
import cv2
import math
import time
import torch
import numpy as np
import random
import argparse
import torchvision
import torch.distributed as dist
from utils.utils import *
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
from collections import OrderedDict
from trainer.dataset.HumanDeictics import Hutics
from trainer.dataset.TEgO import TEgO
import models
# import segmentation_models_pytorch as smp
from models.obj.efficientunet import * 
from utils.ckpt_rgb2bgr import state_dict_rgb2bgr


def unnorm(img, bgr=True):
    if bgr:
        inv_trans = torchvision.transforms.Normalize(
            mean=[-0.406/0.225, -0.456/0.224, -0.485/0.229],
            std=[1/0.225,1/0.224,1/0.229]
        )
    else:
        inv_trans = torchvision.transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229,1/0.224,1/0.225]
        )
    return inv_trans(img)

def test(model, args):
    if args.datasetname == "tego":
        dataset_test = TEgO("Testing", args.dataset, concat_inputs=True, norm=True)
    elif args.datasetname == "hutics":
        dataset_test = Hutics("test", args.dataset, concat_inputs=True, norm=True, relpath=True, bgr=args.bgr)
    dataloader = DataLoader(dataset_test, batch_size=1, pin_memory=True, num_workers=4, shuffle=False)


    model.eval()
    if args.bgr:
        color = np.array([0, 0, 1])
    else:
        color = np.array([1, 0, 0])
    IoU = AverageMeter()
    ioulist = []
    relpathlist = []
    for i, data in tqdm(enumerate(dataloader)):

        inputs, objmask, relpath = data
        inputs = inputs.to(device, non_blocking=True)
        objmask = objmask.to(device, non_blocking=True)

        
        with torch.no_grad():
            pred = model(inputs)
        pred_bi = pred > 0
        iou = torch.logical_and(pred_bi,objmask.bool()).sum() / torch.logical_or(pred_bi,objmask.bool()).sum()
        IoU.update(iou,n=objmask.shape[0])



        img = unnorm(inputs[:,:3], bgr=args.bgr).cpu().detach().numpy()

        pred_bi = pred.cpu().detach().numpy() >0

        # color


        if args.outdir is not None:
            vis = 0.5*img+0.5*pred_bi*color[np.newaxis, :, np.newaxis, np.newaxis]
            # vis = 1*img
            vis = np.transpose((vis[0]*255), (1,2,0)).clip(0,255).astype("uint8")
            savepath = join(args.outdir, relpath[0])
            # print(savepath)
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            if args.bgr:
                cv2.imwrite(savepath, vis)
            else:
                cv2.imwrite(savepath, vis[:,:,::-1])

        ioulist.append(iou)
        relpathlist.append(relpath)
    return IoU.avg, ioulist, relpathlist

def load_model(model, ckptpath):
    state_dict = torch.load(ckptpath)["state_dict"] 
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dataset', type=str, default="/data02/zhongyi/dataset/Deictics/HumanDeictics/",
                        help="dataset dir root path")
    parser.add_argument('--datasetname', type=str, required=True, help="tego or hutics")
    # parser.add_argument('--logdir', required=False, default=None,help="logdir")
    parser.add_argument('--outdir', required=False, default=None,help="")
    parser.add_argument('--bgr', action="store_true")
    parser.add_argument('--modelname', default="unknown", help="")
    parser.add_argument('--modelpath', type=str, default=None, help="model path")

    

    args = parser.parse_args()
    print(args)

    encoder_weights = "imagenet"
    device = torch.device("cuda", 0)
    if not args.modelname or args.modelname.split('_')[0]=="unet":
        model = smp.Unet(encoder_name=f"efficientnet-{args.modelname.split('_')[1]}", encoder_weights=encoder_weights, in_channels=4, classes=1)   
    elif args.modelname.split('_')[0] == "unet++":
        model = smp.UnetPlusPlus(encoder_name=f"efficientnet-{args.modelname.split('_')[1]}", encoder_weights=encoder_weights, in_channels=4, classes=1)   
    elif args.modelname.split('_')[0] == "deeplab":
        model = smp.DeepLabV3(encoder_name=f"efficientnet-{args.modelname.split('_')[1]}", encoder_weights=encoder_weights, in_channels=4, classes=1)   
    elif args.modelname.split('_')[0] == "deeplab+":
        model = smp.DeepLabV3Plus(encoder_name=f"efficientnet-{args.modelname.split('_')[1]}", encoder_weights=encoder_weights, in_channels=4, classes=1)     
    

    model = load_model(model=model, ckptpath=args.modelpath)
    model.to(device)

    iou, iou_list, relpathlist = test(model, args)
    print(iou)

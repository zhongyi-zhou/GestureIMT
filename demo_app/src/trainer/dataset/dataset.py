import cv2
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import os
import glob
from os.path import join
import json
import torchvision
# from utils.utils import *
import torchvision.transforms.functional as F
from src.utils.utils import *


class TeachDataset(Dataset):
    def __init__(self, data_dir, include_seg=True, bgr=True, require_norm=True, splitjson=None, settype="train") -> None:
        super().__init__()
        self.bgr = bgr

        self.require_norm = require_norm
        self.seg = include_seg
        if splitjson is None:
            self.paths = self.init_paths(datadir=data_dir)
        else:
            self.paths = self.init_path_by_split(datadir=data_dir, splitlist=splitjson[settype])

    def init_path_by_split(self, datadir, splitlist):
        with open(join(datadir, "label.txt"), "r") as f:
            lines = f.readlines()
        labels = [line.strip("\n").split(",")[1] for line in lines]

        paths = []
        for id in splitlist:
            label = int(labels[id])
            imgpath = join(datadir, "dataset/img", convert_n_digit(id, 4)+".jpg")
            if self.seg:
                maskpath =  join(datadir, "dataset/mask", convert_n_digit(id, 4)+".png")
                paths.append([imgpath, label, maskpath])
                continue
            paths.append([imgpath, label])

        return paths
    
    def init_paths(self, datadir):
        imgpaths = glob.glob(join(datadir, "dataset/img", "*.jpg"))
        imgpaths.sort()
        if self.seg:
            maskpaths =  glob.glob(join(datadir, "dataset/mask", "*.png"))
            maskpaths.sort()
        with open(join(datadir, "label.txt"), "r") as f:
            lines = f.readlines()
        labels = [line.strip("\n").split(",")[1] for line in lines]

        paths = []
        for i, label in enumerate(labels):
            label = int(label)
            if self.seg:
                paths.append([imgpaths[i], label, maskpaths[i]])
            else:
                paths.append([imgpaths[i], label])
        return paths
    
    def __len__(self):
        return self.paths.__len__()
    
    def getimg(self, index):
        if self.seg:
            imgpath, label, maskpath = self.paths[index]
            img = cv2.imread(imgpath)/255.
            if not self.bgr:
                img = img[:,:,::-1]
            mask = cv2.imread(maskpath, 0)/255. * (label+1)
            return img, label, mask.astype("int")
        else:
            imgpath, label = self.paths[index]
            img = cv2.imread(imgpath)/255.
            if not self.bgr:
                img = img[:,:,::-1]
            return img, label

    def __getitem__(self, index):
        if self.seg:
            img, label, mask = self.getimg(index)
            img = torch.from_numpy(img.copy()).permute(2,0,1).float()
            if self.require_norm:
                if self.bgr:
                    trans = torchvision.transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
                else:
                    trans = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                img = trans(img)
            mask = torch.from_numpy(mask.copy()).long()
            label = torch.tensor(label)
            return img, label, mask
        else:
            img, label = self.getimg(index)
            img = torch.from_numpy(img.copy()).permute(2,0,1).float()
            if self.require_norm:
                if self.bgr:
                    trans = torchvision.transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
                else:
                    trans = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                img = trans(img)
            label = torch.tensor(label)
            return img, label

class MyDataset(Dataset):
    def __init__(self, include_seg=True) -> None:
        super().__init__()
        self.seg = include_seg
        self.data = []
    
    def add_data(self, data, require_norm=True, bgr=False):
        if self.seg:
            # img, gt, mask 
            if data.__len__() != 3:
                raise ValueError(f"The new data shape should have 3 elements but got {data.__len__()}")
        else:
            # img, gt
            if data.__len__() != 2:
                raise ValueError(f"The new data shape should have 2 elements but got {data.__len__()}")
        
        img, gt = data[:2]
        if self.seg:
            mask = data[2]
            mask = torch.from_numpy((mask / 255. * (gt+1)).astype("int").copy()).long()
        if bgr:
            img = img[:,:,::-1]
        img = torch.from_numpy(img.copy()/255.).permute(2,0,1).float()
        gt = torch.tensor(gt)

        if require_norm:
            # rgb
            trans = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img = trans(img)

        if self.seg:
            self.data.append([img, gt, mask])
        else:
            self.data.append([img, gt])
    
    def __len__(self):
        return self.data.__len__()
    
    def __getitem__(self, index):
        return self.data[index]


